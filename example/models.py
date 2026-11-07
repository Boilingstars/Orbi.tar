"""Sequential dispatcher and the same decoder under CTDE.

One encoder reads the fleet and the job slots. Satellites then pick a slot,
idle, or calibrate in index order. A taken slot leaves the mask, and a third
downlink is closed. The log-probability is the sum of those conditional choices.
CTDE keeps this decoder and swaps the encoder for a local one; the critic
still reads the whole fleet.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from sb3_contrib.common.maskable.distributions import MaskableCategorical
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from orbi.env import EDGE_FEATURES, GLOBAL_FEATURES, JOB_FEATURES, NEIGHBOR_FEATURES, SAT_FEATURES


# Negative infinity for masked logits.
_NEG = -1e9

# Index of the presence feature.
_PRESENCE = 11

# Index of the downlink feature.
_DOWNLINK = 8

# Maximum number of downlinks per satellite.
_DOWNLINK_CAP = 2


class OpsAttentionNet(nn.Module):
    """Central encoder plus the sequential decoder. `forward` returns per-satellite logits and a value."""

    def __init__(self, sat_dim=SAT_FEATURES, job_dim=JOB_FEATURES, edge_dim=EDGE_FEATURES, global_dim=GLOBAL_FEATURES, d=128, heads=4, blocks=2, ff=512, d_pointer=64):
        super().__init__()
        self.encoder = _Encoder(sat_dim, job_dim, global_dim, d, heads, blocks, ff, local=False)
        self.pointer = _Pointer(d, edge_dim, d_pointer)

    @property
    def value_head(self):
        return self.encoder.value_head

    @property
    def pool_sat(self):
        return self.encoder.pool_sat

    @property
    def pool_job(self):
        return self.encoder.pool_job

    @property
    def pool_plan(self):
        return self.encoder.pool_plan

    def value(self, obs):
        return self.encoder(obs)[-1]

    def _run(self, obs, masks=None, deterministic=False, actions=None):
        h_s, h_j, edges, jobs, value = self.encoder(obs)
        picked, log_prob, entropy, logits = _sequence(self.pointer, h_s, h_j, edges, jobs, masks, deterministic, actions)
        return picked, log_prob, entropy, logits, value

    def forward(self, obs):
        _, _, _, logits, value = self._run(obs, deterministic=True)
        return logits.flatten(1), value

    def act(self, obs, masks, deterministic=False):
        actions, log_prob, _, _, value = self._run(obs, masks, deterministic)
        return actions, value, log_prob

    def evaluate(self, obs, actions, masks):
        _, log_prob, entropy, _, value = self._run(obs, masks, actions=actions)
        return value, log_prob, entropy


class LocalActor(nn.Module):
    """Same decoder. Each satellite sees its row, the slots, its edges, and its neighbors."""

    def __init__(self, sat_dim=SAT_FEATURES, job_dim=JOB_FEATURES, edge_dim=EDGE_FEATURES, global_dim=GLOBAL_FEATURES, neighbor_dim=NEIGHBOR_FEATURES, d=128, heads=4, blocks=2, ff=512, d_pointer=64):
        super().__init__()
        self.encoder = _Encoder(
            sat_dim, job_dim, global_dim, d, heads, blocks, ff, local=True, neighbor_dim=neighbor_dim
        )
        self.pointer = _Pointer(d, edge_dim, d_pointer)

    def _run(self, obs, masks=None, deterministic=False, actions=None):
        h_s, h_j, edges, jobs, _ = self.encoder(obs)
        return _sequence(self.pointer, h_s, h_j, edges, jobs, masks, deterministic, actions)

    def forward(self, obs):
        return self._run(obs, deterministic=True)[3].flatten(1)

    def act(self, obs, masks, deterministic=False):
        actions, log_prob, _, _ = self._run(obs, masks, deterministic)
        return actions, log_prob

    def evaluate(self, obs, actions, masks):
        _, log_prob, entropy, _ = self._run(obs, masks, actions=actions)
        return log_prob, entropy


class CtdeNet(nn.Module):
    """Local sequential actor. The critic is the fleet encoder and does not pick actions."""

    def __init__(self, neighbor_dim=NEIGHBOR_FEATURES, **kwargs):
        super().__init__()
        self.critic = _Encoder(
            kwargs.get("sat_dim", SAT_FEATURES),
            kwargs.get("job_dim", JOB_FEATURES),
            kwargs.get("global_dim", GLOBAL_FEATURES),
            kwargs.get("d", 128),
            kwargs.get("heads", 4),
            kwargs.get("blocks", 2),
            kwargs.get("ff", 512),
            local=False,
        )
        self.actor = LocalActor(neighbor_dim=neighbor_dim, **kwargs)

    def value(self, obs):
        return self.critic(obs)[-1]

    def forward(self, obs):
        return self.actor(obs), self.value(obs)

    def act(self, obs, masks, deterministic=False):
        actions, log_prob = self.actor.act(obs, masks, deterministic)
        return actions, self.value(obs), log_prob

    def evaluate(self, obs, actions, masks):
        log_prob, entropy = self.actor.evaluate(obs, actions, masks)
        return self.value(obs), log_prob, entropy


class _NullExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=1)

    def forward(self, observations):
        raise RuntimeError("the policy reads the Dict observation directly")


class _IdentityExtractor(nn.Module):
    latent_dim_pi = 1
    latent_dim_vf = 1

    def forward(self, features):
        return features, features


class OpsAttentionPolicy(MaskableActorCriticPolicy):
    """MaskablePPO policy. `controller` is `dispatcher` or `ctde`; both decode satellites in order."""

    def __init__(self, observation_space, action_space, lr_schedule, d=128, heads=4, blocks=2, ff=512, d_pointer=64, controller="dispatcher", **kwargs):
        self._controller = controller
        self._net_cfg = dict(d=d, heads=heads, blocks=blocks, ff=ff, d_pointer=d_pointer)
        kwargs.update(net_arch=[], ortho_init=False, features_extractor_class=_NullExtractor)
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = _IdentityExtractor()

    def _build(self, lr_schedule):
        space = self.observation_space
        cfg = dict(
            sat_dim=space["sats"].shape[-1],
            job_dim=space["jobs"].shape[-1],
            edge_dim=space["edges"].shape[-1],
            global_dim=space["global"].shape[0],
            **self._net_cfg,
        )
        if self._controller == "ctde":
            neighbor_dim = space["neighbors"].shape[-1] if "neighbors" in space.spaces else NEIGHBOR_FEATURES
            self.net = CtdeNet(neighbor_dim=neighbor_dim, **cfg)
        else:
            self.net = OpsAttentionNet(**cfg)
        self.action_net = nn.Identity()
        self.value_net = nn.Identity()
        self.mlp_extractor = _IdentityExtractor()
        self.optimizer = self.optimizer_class(self.net.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def forward(self, obs, deterministic=False, action_masks=None):
        return self.net.act(obs, action_masks, deterministic)

    def evaluate_actions(self, obs, actions, action_masks=None):
        return self.net.evaluate(obs, actions, action_masks)

    def predict_values(self, obs):
        return self.net.value(obs)

    def _predict(self, observation, deterministic=False, action_masks=None):
        return self.net.act(observation, action_masks, deterministic)[0]


def count_parameters(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def _open(allowed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep a row with no legal key from a NaN softmax. `gate` drops that row's output."""
    alive = allowed.any(dim=-1, keepdim=True)
    return allowed | ~alive, alive


class _Attention(nn.Module):
    def __init__(self, d: int, heads: int, self_attention: bool) -> None:
        super().__init__()
        if d % heads:
            raise ValueError("d must be divisible by the number of heads")
        self.heads = heads
        self.head_dim = d // heads
        self.self_attention = self_attention
        self.qkv = nn.Linear(d, 3 * d) if self_attention else None
        self.q_proj = None if self_attention else nn.Linear(d, d)
        self.kv_proj = None if self_attention else nn.Linear(d, 2 * d)
        self.out = nn.Linear(d, d)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, length, _ = x.shape
        return x.view(b, length, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.self_attention:
            query, key, value = (self._split(t) for t in self.qkv(q).chunk(3, dim=-1))
        else:
            query = self._split(self.q_proj(q))
            key, value = (self._split(t) for t in self.kv_proj(kv).chunk(2, dim=-1))
        out = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
        b, _, length, _ = out.shape
        return self.out(out.transpose(1, 2).reshape(b, length, self.heads * self.head_dim))


class _Block(nn.Module):
    def __init__(self, d: int, heads: int, ff: int, self_attention: bool) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d)
        self.norm_kv = None if self_attention else nn.LayerNorm(d)
        self.attn = _Attention(d, heads, self_attention)
        self.norm_ff = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Linear(ff, d))

    def forward(self, q, kv, mask, gate=None):
        h_q = self.norm_q(q)
        h_kv = h_q if self.norm_kv is None else self.norm_kv(kv)
        out = self.attn(h_q, h_kv, mask)
        if gate is not None:
            out = out * gate
        x = q + out
        return x + self.ff(self.norm_ff(x))


def _pool(h, valid, scorer):
    any_valid = valid.any(dim=1, keepdim=True)
    opened = valid.clone()
    opened[:, 0] |= ~any_valid.squeeze(1)
    weights = torch.softmax(scorer(h).squeeze(-1).masked_fill(~opened, _NEG), dim=1).unsqueeze(-1)
    return (h * weights).sum(dim=1) * any_valid.float()


class _Encoder(nn.Module):
    """Fleet/job encoder. `local` drops fleet self-attention and reads neighbors instead."""

    def __init__(self, sat_dim, job_dim, global_dim, d, heads, blocks, ff, local=False, neighbor_dim=NEIGHBOR_FEATURES):
        super().__init__()
        self.local = local
        self.blocks = blocks
        self.emb_sat = nn.Sequential(nn.Linear(sat_dim, d), nn.LayerNorm(d), nn.ReLU())
        self.emb_job = nn.Sequential(nn.Linear(job_dim, d), nn.LayerNorm(d), nn.ReLU())
        self.emb_global = nn.Sequential(nn.Linear(global_dim, d), nn.LayerNorm(d), nn.ReLU())
        self.job_self = nn.ModuleList(_Block(d, heads, ff, True) for _ in range(blocks))
        self.sat_cross = nn.ModuleList(_Block(d, heads, ff, False) for _ in range(blocks))
        if local:
            self.emb_neighbor = nn.Sequential(nn.Linear(neighbor_dim, d), nn.LayerNorm(d), nn.ReLU())
            self.pool_neighbor = nn.Linear(d, 1)
            self.sat_self = self.job_cross = None
        else:
            self.sat_self = nn.ModuleList(_Block(d, heads, ff, True) for _ in range(blocks))
            self.job_cross = nn.ModuleList(_Block(d, heads, ff, False) for _ in range(blocks))
            self.pool_sat = nn.Linear(d, 1)
            self.pool_job = nn.Linear(d, 1)
            self.pool_plan = nn.Linear(d, 1)
            self.value_head = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, obs):
        # Tensors for this tick. A local actor does not see fleet SoC.
        sats = obs["sats"].float()
        jobs = obs["jobs"].float()
        edges = obs["edges"].float()
        glob = obs["global"].float()
        if self.local:
            glob = glob.clone()
            glob[..., 4:6] = 0.0
        b = sats.shape[0]
        sat_valid = sats[..., _PRESENCE] > 0
        job_valid = jobs[..., 0:3].sum(-1) > 0
        eligible = edges[..., 0] > 0

        # Embed satellites and slots, and fold the whole plan into the global vector.
        h_g = self.emb_global(glob)
        if not self.local and "plan" in obs:
            plan = obs["plan"].float()
            h_g = h_g + _pool(self.emb_job(plan), plan[..., 0:3].sum(-1) > 0, self.pool_plan)
        h_s = self.emb_sat(sats) + h_g.unsqueeze(1)
        h_j = self.emb_job(jobs) + h_g.unsqueeze(1)

        # On-board actors pool their neighbor memory instead of attending the fleet.
        if self.local:
            neighbors = obs["neighbors"].float()
            nb_valid = neighbors[..., 0] > 0
            h_nb = self.emb_neighbor(neighbors)
            any_nb = nb_valid.any(-1, keepdim=True)
            opened = nb_valid.clone()
            opened[..., 0] |= ~any_nb.squeeze(-1)
            weights = torch.softmax(self.pool_neighbor(h_nb).squeeze(-1).masked_fill(~opened, _NEG), dim=-1)
            h_s = h_s + (h_nb * weights.unsqueeze(-1)).sum(2) * any_nb.float()

        # Keys that exist, then one block of self-attention and cross-attention.
        sat_keys, sat_alive = _open(sat_valid[:, None, None, :])
        job_keys, job_alive = _open(job_valid[:, None, None, :])
        cross, row_alive = _open((eligible & job_valid.unsqueeze(1) & sat_valid.unsqueeze(2)).unsqueeze(1))
        sat_gate = sat_alive.view(b, 1, 1).float()
        job_gate = job_alive.view(b, 1, 1).float()
        row_gate = row_alive.squeeze(1).float()
        for i in range(self.blocks):
            if self.sat_self is not None:
                h_s = self.sat_self[i](h_s, h_s, sat_keys, sat_gate)
            h_j = self.job_self[i](h_j, h_j, job_keys, job_gate)
            new_s = self.sat_cross[i](h_s, h_j, cross, row_gate)
            if self.job_cross is not None:
                back, col_alive = _open((cross.squeeze(1).transpose(1, 2) & sat_valid.unsqueeze(1)).unsqueeze(1))
                h_j = self.job_cross[i](h_j, h_s, back, col_alive.squeeze(1).float())
            h_s = new_s

        # The critic reads both pools. A local encoder has no value head.
        value = None
        if not self.local:
            value = self.value_head(torch.cat([_pool(h_s, sat_valid, self.pool_sat), _pool(h_j, job_valid, self.pool_job), h_g], -1))
        return h_s, h_j, edges, jobs, value


class _Pointer(nn.Module):
    def __init__(self, d, edge_dim, d_pointer):
        super().__init__()
        self.q = nn.Linear(d, d_pointer, bias=False)
        self.k = nn.Linear(d, d_pointer, bias=False)
        self.e = nn.Linear(edge_dim, d_pointer, bias=False)
        self.v = nn.Linear(d_pointer, 1, bias=False)
        self.idle = nn.Linear(d, 1)
        self.calibrate = nn.Linear(d, 1)

    def logits(self, h_s_i, h_j, edge_i):
        pair = self.q(h_s_i).unsqueeze(1) + self.k(h_j) + self.e(edge_i)
        scores = self.v(torch.tanh(pair)).squeeze(-1)
        return torch.cat([scores, self.idle(h_s_i), self.calibrate(h_s_i)], -1)


def _mask_tensor(masks, n, k, device):
    if masks is None:
        return None
    return torch.as_tensor(masks, device=device, dtype=torch.bool).reshape(-1, n, k + 2)


def _sequence(pointer, h_s, h_j, edges, jobs, masks, deterministic=False, actions=None):
    """Pick in index order. Each step sees slots already taken and the downlink cap."""
    b, n, _ = h_s.shape
    k = h_j.shape[1]
    device = h_s.device
    base = _mask_tensor(masks, n, k, device)

    # Slots already taken, and how many downlinks earlier satellites opened.
    job_valid = jobs[..., 0:3].sum(-1) > 0
    is_down = jobs[..., _DOWNLINK] > 0.5
    taken = torch.zeros(b, k, dtype=torch.bool, device=device)
    downs = torch.zeros(b, dtype=torch.long, device=device)
    chosen, lps, ents, rows = [], [], [], []

    for i in range(n):
        # This satellite's legal set: the env mask, minus taken slots and a full downlink cap.
        logits = pointer.logits(h_s[:, i], h_j, edges[:, i])

        if base is None:
            legal = torch.ones(b, k + 2, dtype=torch.bool, device=device)
        else:
            legal = base[:, i].clone()

        legal[:, :k] &= job_valid & ~taken
        room = (downs < _DOWNLINK_CAP).unsqueeze(1)
        legal[:, :k] &= ~is_down | room
        legal[torch.arange(b, device=device)[~legal.any(-1)], k] = True
        dist = MaskableCategorical(logits=logits, masks=legal)

        if actions is not None:
            pick = actions[:, i].long()
        elif deterministic:
            pick = logits.masked_fill(~legal, _NEG).argmax(-1)
        else:
            pick = dist.sample()

        chosen.append(pick)
        lps.append(dist.log_prob(pick))
        ents.append(dist.entropy())
        rows.append(logits.masked_fill(~legal, _NEG))

        # A job slot closes. Idle and calibrate leave it for the next satellite.
        job_pick = pick < k
        took = torch.zeros(b, k, dtype=torch.bool, device=device)
        took[torch.arange(b, device=device), torch.where(job_pick, pick, torch.zeros_like(pick))] = job_pick

        taken = taken | took
        downs = downs + (took & is_down).sum(-1)
    return torch.stack(chosen, 1), torch.stack(lps, 1).sum(1), torch.stack(ents, 1).sum(1), torch.stack(rows, 1)


__all__ = ["CtdeNet", "LocalActor", "OpsAttentionNet", "OpsAttentionPolicy", "count_parameters"]
