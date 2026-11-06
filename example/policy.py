"""Attention-политика диспетчера: один forward на тик, матрица оценок N x (K+2).

Сеть не зависит от числа аппаратов и заданий: `N` и `K` живут только в
размерностях тензоров, все веса общие. Поэтому модель, обученная на P02 (48 КА),
запускается на P01 (16 КА) без переобучения — лишние строки закрыты padding-масками.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from sb3_contrib.common.maskable.distributions import MaskableMultiCategoricalDistribution
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from orbi.env import (
    EDGE_FEATURES,
    GLOBAL_FEATURES,
    JOB_FEATURES,
    NEIGHBOR_FEATURES,
    SAT_FEATURES,
)

_NEG = -1e9

# Признак «аппарат существует». `validate()` организаторов требует
# `capacity_wh > 0` у каждого КА, а `OpsEnv` заполняет padding-строки нулями,
# поэтому этот столбец — однозначный индикатор присутствия без 15-го признака.
_PRESENCE_FEATURE = 11


def _open_degenerate(allowed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Строки без единого разрешённого ключа: softmax по пустому множеству даёт NaN.

    Открываем такой строке первый ключ и гасим её выход множителем `gate`,
    так что в residual остаётся только исходное состояние.
    """
    alive = allowed.any(dim=-1, keepdim=True)
    return allowed | ~alive, alive


class _Attention(nn.Module):
    """Многоголовое внимание на `scaled_dot_product_attention`.

    Слитая проекция QKV и маска, широковещательная по головам: у `nn.MultiheadAttention`
    маска приходится раздувать до `B*heads x L x S`, а это на длинном rollout
    заметно дороже самого внимания.
    """

    def __init__(self, d: int, heads: int, self_attention: bool) -> None:
        super().__init__()
        if d % heads:
            raise ValueError("d должно делиться на число голов")
        self.heads = heads
        self.head_dim = d // heads
        self.self_attention = self_attention
        if self_attention:
            self.qkv = nn.Linear(d, 3 * d)
        else:
            self.q_proj = nn.Linear(d, d)
            self.kv_proj = nn.Linear(d, 2 * d)
        self.out = nn.Linear(d, d)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, length, _ = x.shape
        return x.view(b, length, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.self_attention:
            qkv = self.qkv(q).chunk(3, dim=-1)
            query, key, value = (self._split(t) for t in qkv)
        else:
            query = self._split(self.q_proj(q))
            key, value = (self._split(t) for t in self.kv_proj(kv).chunk(2, dim=-1))
        out = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
        b, _, length, _ = out.shape
        return self.out(out.transpose(1, 2).reshape(b, length, self.heads * self.head_dim))


class _Block(nn.Module):
    """Pre-LN блок внимания с FFN — стандартный трансформерный слой."""

    def __init__(self, d: int, heads: int, ff: int, self_attention: bool) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d) if not self_attention else None
        self.attn = _Attention(d, heads, self_attention)
        self.norm_ff = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Linear(ff, d))

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        mask: torch.Tensor,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h_q = self.norm_q(q)
        h_kv = h_q if self.norm_kv is None else self.norm_kv(kv)
        out = self.attn(h_q, h_kv, mask)
        if gate is not None:
            out = out * gate
        x = q + out
        return x + self.ff(self.norm_ff(x))


class OpsAttentionNet(nn.Module):
    """Энкодер группировки и очереди заданий, общий для актора и критика."""

    def __init__(
        self,
        sat_dim: int = SAT_FEATURES,
        job_dim: int = JOB_FEATURES,
        edge_dim: int = EDGE_FEATURES,
        global_dim: int = GLOBAL_FEATURES,
        d: int = 128,
        heads: int = 4,
        blocks: int = 2,
        ff: int = 512,
        d_pointer: int = 64,
    ) -> None:
        super().__init__()
        self.d = d
        self.heads = heads
        self.blocks = blocks

        self.emb_sat = nn.Sequential(nn.Linear(sat_dim, d), nn.LayerNorm(d), nn.ReLU())
        self.emb_job = nn.Sequential(nn.Linear(job_dim, d), nn.LayerNorm(d), nn.ReLU())
        self.emb_global = nn.Sequential(nn.Linear(global_dim, d), nn.LayerNorm(d), nn.ReLU())

        self.sat_self = nn.ModuleList(_Block(d, heads, ff, True) for _ in range(blocks))
        self.job_self = nn.ModuleList(_Block(d, heads, ff, True) for _ in range(blocks))
        self.sat_cross = nn.ModuleList(_Block(d, heads, ff, False) for _ in range(blocks))
        self.job_cross = nn.ModuleList(_Block(d, heads, ff, False) for _ in range(blocks))

        self.ptr_q = nn.Linear(d, d_pointer, bias=False)
        self.ptr_k = nn.Linear(d, d_pointer, bias=False)
        self.ptr_e = nn.Linear(edge_dim, d_pointer, bias=False)
        self.ptr_v = nn.Linear(d_pointer, 1, bias=False)

        self.head_idle = nn.Linear(d, 1)
        self.head_calibrate = nn.Linear(d, 1)

        self.pool_sat = nn.Linear(d, 1)
        self.pool_job = nn.Linear(d, 1)
        self.pool_plan = nn.Linear(d, 1)
        self.value_head = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(), nn.Linear(d, 1))

    @staticmethod
    def _pool(h: torch.Tensor, valid: torch.Tensor, scorer: nn.Linear) -> torch.Tensor:
        scores = scorer(h).squeeze(-1).masked_fill(~valid, _NEG)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (h * weights).sum(dim=1) * valid.any(dim=1, keepdim=True).float()

    @staticmethod
    def _pool_masked(h: torch.Tensor, valid: torch.Tensor, scorer: nn.Linear) -> torch.Tensor:
        """Пул, у которого часть строк — padding. Пустой набор даёт нулевой вектор, не NaN."""
        any_valid = valid.any(dim=1, keepdim=True)
        opened = valid.clone()
        opened[:, 0] |= ~any_valid.squeeze(1)
        scores = scorer(h).squeeze(-1).masked_fill(~opened, _NEG)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (h * weights).sum(dim=1) * any_valid.float()

    def forward(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        sats = obs["sats"].float()
        jobs = obs["jobs"].float()
        edges = obs["edges"].float()
        glob = obs["global"].float()
        b, n, _ = sats.shape
        k = jobs.shape[1]

        sat_valid = sats[..., _PRESENCE_FEATURE] > 0
        job_valid = jobs[..., 0:3].sum(dim=-1) > 0
        eligible = edges[..., 0] > 0

        h_g = self.emb_global(glob)
        # Весь плановый список, включая ещё не открытые задания. Attention по
        # 8192 строкам не считается: эмбеддинг общий со слотами действия, дальше
        # только пул в глобальный вектор.
        if "plan" in obs:
            plan = obs["plan"].float()
            plan_valid = plan[..., 0:3].sum(dim=-1) > 0
            h_g = h_g + self._pool_masked(self.emb_job(plan), plan_valid, self.pool_plan)
        h_s = self.emb_sat(sats) + h_g.unsqueeze(1)
        h_j = self.emb_job(jobs) + h_g.unsqueeze(1)

        # Маски хранятся как (B, 1, L, S) и транслируются по головам внутри SDPA.
        sat_keys, sat_alive = _open_degenerate(sat_valid[:, None, None, :])
        job_keys, job_alive = _open_degenerate(job_valid[:, None, None, :])
        sat_gate = sat_alive.view(b, 1, 1).float()
        job_gate = job_alive.view(b, 1, 1).float()

        # Кто с кем может работать: eligibility — свойство пары, поэтому она
        # задаёт маску кросс-внимания, а не признак одного объекта.
        cross_sj, row_alive = _open_degenerate(
            (eligible & job_valid.unsqueeze(1)).unsqueeze(1)
        )
        cross_js, col_alive = _open_degenerate(
            (cross_sj.squeeze(1).transpose(1, 2) & sat_valid.unsqueeze(1)).unsqueeze(1)
        )
        gate_s = row_alive.squeeze(1).float()
        gate_j = col_alive.squeeze(1).float()

        for i in range(self.blocks):
            h_s = self.sat_self[i](h_s, h_s, sat_keys, gate=sat_gate)
            h_j = self.job_self[i](h_j, h_j, job_keys, gate=job_gate)
            new_s = self.sat_cross[i](h_s, h_j, cross_sj, gate=gate_s)
            new_j = self.job_cross[i](h_j, h_s, cross_js, gate=gate_j)
            h_s, h_j = new_s, new_j

        z_s, z_j = h_s, h_j

        # Pointer: сравниваются пары «аппарат — задание», поэтому число заданий
        # может меняться от тика к тику без смены весов. tanh на месте — самый
        # крупный тензор прохода, B x N x K x d_pointer, лишней копии не нужно.
        pair = self.ptr_q(z_s).unsqueeze(2) + self.ptr_k(z_j).unsqueeze(1) + self.ptr_e(edges)
        scores = self.ptr_v(torch.tanh_(pair)).squeeze(-1)

        logits = torch.cat(
            [scores, self.head_idle(z_s), self.head_calibrate(z_s)], dim=-1
        ).reshape(b, n * (k + 2))

        value = self.value_head(
            torch.cat(
                [
                    self._pool(z_s, sat_valid, self.pool_sat),
                    self._pool(z_j, job_valid, self.pool_job),
                    h_g,
                ],
                dim=-1,
            )
        )
        return logits, value


class LocalActor(nn.Module):
    """Shared on-board actor: own row, jobs, own edges, neighbors. No fleet self-attention."""

    def __init__(
        self,
        sat_dim: int = SAT_FEATURES,
        job_dim: int = JOB_FEATURES,
        edge_dim: int = EDGE_FEATURES,
        global_dim: int = GLOBAL_FEATURES,
        neighbor_dim: int = NEIGHBOR_FEATURES,
        d: int = 128,
        heads: int = 4,
        blocks: int = 2,
        ff: int = 512,
        d_pointer: int = 64,
    ) -> None:
        super().__init__()
        self.d = d
        self.heads = heads
        self.blocks = blocks
        self.emb_sat = nn.Sequential(nn.Linear(sat_dim, d), nn.LayerNorm(d), nn.ReLU())
        self.emb_job = nn.Sequential(nn.Linear(job_dim, d), nn.LayerNorm(d), nn.ReLU())
        self.emb_global = nn.Sequential(nn.Linear(global_dim, d), nn.LayerNorm(d), nn.ReLU())
        self.emb_neighbor = nn.Sequential(nn.Linear(neighbor_dim, d), nn.LayerNorm(d), nn.ReLU())
        self.pool_neighbor = nn.Linear(d, 1)
        self.job_self = nn.ModuleList(_Block(d, heads, ff, True) for _ in range(blocks))
        self.sat_cross = nn.ModuleList(_Block(d, heads, ff, False) for _ in range(blocks))
        self.ptr_q = nn.Linear(d, d_pointer, bias=False)
        self.ptr_k = nn.Linear(d, d_pointer, bias=False)
        self.ptr_e = nn.Linear(edge_dim, d_pointer, bias=False)
        self.ptr_v = nn.Linear(d_pointer, 1, bias=False)
        self.head_idle = nn.Linear(d, 1)
        self.head_calibrate = nn.Linear(d, 1)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        sats = obs["sats"].float()
        jobs = obs["jobs"].float()
        edges = obs["edges"].float()
        glob = obs["global"].float().clone()
        glob[..., 4:6] = 0.0
        b, n, _ = sats.shape
        k = jobs.shape[1]
        neighbors = obs["neighbors"].float()

        sat_valid = sats[..., _PRESENCE_FEATURE] > 0
        job_valid = jobs[..., 0:3].sum(dim=-1) > 0
        eligible = edges[..., 0] > 0
        nb_valid = neighbors[..., 0] > 0

        h_g = self.emb_global(glob)
        h_s = self.emb_sat(sats) + h_g.unsqueeze(1)
        h_j = self.emb_job(jobs) + h_g.unsqueeze(1)

        h_nb = self.emb_neighbor(neighbors)
        any_nb = nb_valid.any(dim=-1, keepdim=True)
        opened = nb_valid.clone()
        opened[..., 0] |= ~any_nb.squeeze(-1)
        nb_scores = self.pool_neighbor(h_nb).squeeze(-1).masked_fill(~opened, _NEG)
        nb_w = torch.softmax(nb_scores, dim=-1).unsqueeze(-1)
        h_s = h_s + (h_nb * nb_w).sum(dim=2) * any_nb.float()

        job_keys, job_alive = _open_degenerate(job_valid[:, None, None, :])
        job_gate = job_alive.view(b, 1, 1).float()
        cross_sj, row_alive = _open_degenerate(
            (eligible & job_valid.unsqueeze(1) & sat_valid.unsqueeze(2)).unsqueeze(1)
        )
        gate_s = row_alive.squeeze(1).float()

        for i in range(self.blocks):
            h_j = self.job_self[i](h_j, h_j, job_keys, gate=job_gate)
            h_s = self.sat_cross[i](h_s, h_j, cross_sj, gate=gate_s)

        pair = self.ptr_q(h_s).unsqueeze(2) + self.ptr_k(h_j).unsqueeze(1) + self.ptr_e(edges)
        scores = self.ptr_v(torch.tanh_(pair)).squeeze(-1)
        return torch.cat(
            [scores, self.head_idle(h_s), self.head_calibrate(h_s)], dim=-1
        ).reshape(b, n * (k + 2))


class CtdeNet(nn.Module):
    """Local shared actor plus a centralized critic on the global observation."""

    def __init__(self, neighbor_dim: int = NEIGHBOR_FEATURES, **kwargs) -> None:
        super().__init__()
        self.critic = OpsAttentionNet(**kwargs)
        self.actor = LocalActor(neighbor_dim=neighbor_dim, **kwargs)
        self.value_head = self.critic.value_head
        self.pool_sat = self.critic.pool_sat
        self.pool_job = self.critic.pool_job
        self.pool_plan = self.critic.pool_plan

    def forward(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        _, value = self.critic(obs)
        return self.actor(obs), value


class _NullExtractor(BaseFeaturesExtractor):
    """Заглушка: сеть читает Dict-наблюдение напрямую, плоские признаки не нужны."""

    def __init__(self, observation_space) -> None:
        super().__init__(observation_space, features_dim=1)

    def forward(self, observations):  # pragma: no cover - не вызывается
        raise RuntimeError("OpsAttentionPolicy работает с Dict-наблюдением напрямую")


class _IdentityExtractor(nn.Module):
    latent_dim_pi = 1
    latent_dim_vf = 1

    def forward(self, features):  # pragma: no cover - не вызывается
        return features, features


class OpsAttentionPolicy(MaskableActorCriticPolicy):
    """MaskablePPO-политика: диспетчер или CTDE (`controller`)."""

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        d: int = 128,
        heads: int = 4,
        blocks: int = 2,
        ff: int = 512,
        d_pointer: int = 64,
        controller: str = "dispatcher",
        **kwargs,
    ) -> None:
        self._controller = controller
        self._net_cfg = dict(d=d, heads=heads, blocks=blocks, ff=ff, d_pointer=d_pointer)
        kwargs["net_arch"] = []
        kwargs["ortho_init"] = False
        kwargs["features_extractor_class"] = _NullExtractor
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = _IdentityExtractor()

    def _build(self, lr_schedule) -> None:
        space = self.observation_space
        net_kwargs = dict(
            sat_dim=space["sats"].shape[-1],
            job_dim=space["jobs"].shape[-1],
            edge_dim=space["edges"].shape[-1],
            global_dim=space["global"].shape[0],
            **self._net_cfg,
        )
        if self._controller == "ctde":
            neighbor_dim = (
                space["neighbors"].shape[-1] if "neighbors" in space.spaces else NEIGHBOR_FEATURES
            )
            self.net = CtdeNet(neighbor_dim=neighbor_dim, **net_kwargs)
        else:
            self.net = OpsAttentionNet(**net_kwargs)
        self.action_dist = MaskableMultiCategoricalDistribution(list(self.action_space.nvec))
        self.action_net = nn.Identity()
        self.value_net = nn.Identity()
        self.mlp_extractor = _IdentityExtractor()
        self.optimizer = self.optimizer_class(
            self.net.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _logits_value(self, obs) -> tuple[torch.Tensor, torch.Tensor]:
        return self.net(obs)

    def _dist(self, logits: torch.Tensor, action_masks=None):
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution

    def forward(self, obs, deterministic: bool = False, action_masks=None):
        logits, values = self._logits_value(obs)
        distribution = self._dist(logits, action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions, values, distribution.log_prob(actions)

    def evaluate_actions(self, obs, actions, action_masks=None):
        logits, values = self._logits_value(obs)
        distribution = self._dist(logits, action_masks)
        return values, distribution.log_prob(actions), distribution.entropy()

    def get_distribution(self, obs, action_masks=None):
        logits, _ = self._logits_value(obs)
        return self._dist(logits, action_masks)

    def predict_values(self, obs):
        _, values = self._logits_value(obs)
        return values

    def _predict(self, observation, deterministic: bool = False, action_masks=None):
        logits, _ = self._logits_value(observation)
        return self._dist(logits, action_masks).get_actions(deterministic=deterministic)


def count_parameters(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


__all__ = ["CtdeNet", "LocalActor", "OpsAttentionNet", "OpsAttentionPolicy", "count_parameters"]
