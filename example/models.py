"""Learned pair scores plus job-first greedy matching.

One encoder reads the fleet and the job slots. A pointer scores every satellite-slot
pair. Matching walks jobs by those scores, assigns at most one free satellite, and
respects the downlink cap. Idle and calibrate are leftover heads. The joint
log-probability is the sum of those conditional choices.
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

from orbi.env import EDGE_FEATURES, GLOBAL_DOWNLINK_CAP, GLOBAL_FEATURES, JOB_FEATURES, N_MAX, NEIGHBOR_FEATURES, SAT_FEATURES


# Negative infinity for masked logits.
_NEG = -1e9

# Index of the presence feature.
_PRESENCE = 11

# Index of the downlink feature.
_DOWNLINK = 8

# Idle and calibrate start this far below a zero slot score, so a fresh policy
# still samples jobs. Skip is the option to leave a slot unassigned.
_ASIDE = -2.0


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
        picked, log_prob, entropy, logits = _match(
            self.pointer, h_s, h_j, edges, jobs, masks, obs, deterministic, actions
        )
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
        return _match(self.pointer, h_s, h_j, edges, jobs, masks, obs, deterministic, actions)

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


def _pool_plan(emb_job, pool_job, plan: torch.Tensor) -> torch.Tensor:
    """Embed only filled plan rows. The env pads the table to PLAN_MAX."""
    valid = plan[..., 0:3].sum(-1) > 0
    filled = valid.any(0)
    last = int(filled.nonzero()[-1].item()) + 1 if filled.any() else 1
    return _pool(emb_job(plan[:, :last]), valid[:, :last], pool_job)


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

        # Embed satellites and slots, and fold unfinished work into the global vector.
        h_g = self.emb_global(glob)
        if not self.local and "plan" in obs:
            h_g = h_g + _pool_plan(self.emb_job, self.pool_plan, obs["plan"].float())
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
        self.skip = nn.Linear(d, 1)

        nn.init.constant_(self.idle.bias, _ASIDE)
        nn.init.constant_(self.calibrate.bias, _ASIDE)
        nn.init.constant_(self.skip.bias, _ASIDE)

    def pair_scores(self, h_s, h_j, edges):
        query = self.q(h_s).unsqueeze(2)
        key = self.k(h_j).unsqueeze(1)
        edge = self.e(edges)
        return self.v(torch.tanh(query + key + edge)).squeeze(-1)

    def side_scores(self, h_s, h_j):
        return self.idle(h_s).squeeze(-1), self.calibrate(h_s).squeeze(-1), self.skip(h_j).squeeze(-1)


def _mask_tensor(masks, n, k, device):
    if masks is None:
        return None
    return torch.as_tensor(masks, device=device, dtype=torch.bool).reshape(-1, n, k + 2)


def _downlink_caps(obs, batch: int, device) -> torch.Tensor:
    """Fleet downlink cap, recovered from global[GLOBAL_DOWNLINK_CAP] = limit / N_MAX."""
    glob = None if not isinstance(obs, dict) else obs.get("global")
    if glob is None:
        return torch.full((batch,), 2, device=device, dtype=torch.long)
    raw = torch.as_tensor(glob, device=device, dtype=torch.float32)
    if raw.ndim == 1:
        raw = raw.unsqueeze(0)
    caps = (raw[:, GLOBAL_DOWNLINK_CAP] * N_MAX).round().clamp(min=1).to(dtype=torch.long)
    if caps.shape[0] == 1 and batch > 1:
        caps = caps.expand(batch)
    return caps


def _match(pointer, h_s, h_j, edges, jobs, masks, obs, deterministic=False, actions=None):
    """Score every pair, then assign jobs in score order. Skip leaves a slot empty."""
    b, n, _ = h_s.shape
    k = h_j.shape[1]
    device = h_s.device
    base = _mask_tensor(masks, n, k, device)
    caps = _downlink_caps(obs, b, device)
    job_valid = jobs[..., 0:3].sum(-1) > 0
    is_down = jobs[..., _DOWNLINK] > 0.5
    sats = torch.as_tensor(obs["sats"], device=device, dtype=torch.float32)
    if sats.ndim == 2:
        sats = sats.unsqueeze(0)
    sat_valid = sats[..., _PRESENCE] > 0

    pair = pointer.pair_scores(h_s, h_j, edges)
    idle, cal, skip = pointer.side_scores(h_s, h_j)
    table = torch.cat([pair, idle.unsqueeze(-1), cal.unsqueeze(-1)], dim=-1)

    legal_pair = pair.clone()
    if base is not None:
        legal_pair = legal_pair.masked_fill(~base[..., :k], _NEG)
    legal_pair = legal_pair.masked_fill(~sat_valid.unsqueeze(-1), _NEG)
    # Learned pair scores pick the job. A light priority nudge, then the pointer.
    prio = jobs[..., 0:3] @ torch.tensor([1.0, 2.0, 3.0], device=device, dtype=pair.dtype)
    job_priority = legal_pair.max(dim=1).values + 0.5 * prio
    job_priority = job_priority.masked_fill(~job_valid, _NEG)
    order = job_priority.argsort(dim=-1, descending=True)

    free = sat_valid.clone()
    downs = torch.zeros(b, dtype=torch.long, device=device)
    chosen = torch.full((b, n), k, dtype=torch.long, device=device)
    lps = []
    ents = []
    rows = torch.full((b, n, k + 2), _NEG, device=device, dtype=table.dtype)
    rows[:, :, k] = idle
    rows[:, :, k + 1] = cal
    rows[:, :, :k] = pair
    batch_idx = torch.arange(b, device=device)

    for step in range(k):
        slot = order[:, step]
        slot_exp = slot.view(b, 1, 1).expand(b, n, 1)
        slot_scores = pair.gather(2, slot_exp).squeeze(-1)
        last_exec = edges[..., 4].gather(2, slot_exp).squeeze(-1)
        slot_scores = slot_scores + 1.0 * sats[..., 0] + last_exec
        slot_skip = skip.gather(1, slot.unsqueeze(1)).squeeze(1)
        slot_down = is_down.gather(1, slot.unsqueeze(1)).squeeze(1)
        slot_ok = job_valid.gather(1, slot.unsqueeze(1)).squeeze(1)
        if base is None:
            sat_ok = sat_valid.clone()
        else:
            sat_ok = base[..., :k].gather(2, slot_exp).squeeze(-1)
        sat_ok = sat_ok & free & slot_ok.unsqueeze(1)
        sat_ok = sat_ok & (~slot_down.unsqueeze(1) | (downs < caps).unsqueeze(1))
        logits = torch.cat([slot_scores, slot_skip.unsqueeze(1)], dim=-1)
        # Sample/eval never skip a legal satellite. Replay must still score a stored skip.
        can_skip = ~sat_ok.any(dim=-1, keepdim=True)
        if actions is not None:
            can_skip = torch.ones_like(can_skip)
        allow, _ = _open(torch.cat([sat_ok, can_skip], dim=-1))
        dist = MaskableCategorical(logits=logits, masks=allow)

        if actions is not None:
            took = actions.long() == slot.unsqueeze(1)
            has = took.any(dim=-1)
            pick = torch.where(has, took.to(torch.long).argmax(dim=-1), torch.full((b,), n, device=device, dtype=torch.long))
        elif deterministic:
            # Inference assigns the best legal satellite. Skip only when the slot is empty.
            best = slot_scores.masked_fill(~sat_ok, _NEG).argmax(-1)
            pick = torch.where(sat_ok.any(dim=-1), best, torch.full((b,), n, device=device, dtype=torch.long))
        else:
            pick = dist.sample()

        lps.append(dist.log_prob(pick))
        ents.append(dist.entropy())
        assigned = pick < n
        if assigned.any():
            chosen[batch_idx[assigned], pick[assigned]] = slot[assigned]
            free = free.clone()
            free[batch_idx[assigned], pick[assigned]] = False
            downs = downs + (assigned & slot_down).long()

    for i in range(n):
        leftover = free[:, i]
        logits_ic = torch.stack([idle[:, i], cal[:, i]], dim=-1)
        allow_ic = torch.ones(b, 2, dtype=torch.bool, device=device)
        if base is not None:
            allow_ic[:, 0] = base[:, i, k]
            allow_ic[:, 1] = base[:, i, k + 1]
        allow_ic, _ = _open(allow_ic)
        dist = MaskableCategorical(logits=logits_ic, masks=allow_ic)
        if actions is not None:
            pick = (actions[:, i].long() == (k + 1)).long()
        elif deterministic:
            pick = logits_ic.masked_fill(~allow_ic, _NEG).argmax(-1)
        else:
            pick = dist.sample()
        lp = dist.log_prob(pick)
        ent = dist.entropy()
        lps.append(torch.where(leftover, lp, torch.zeros_like(lp)))
        ents.append(torch.where(leftover, ent, torch.zeros_like(ent)))
        chosen[:, i] = torch.where(leftover & (pick == 1), torch.full_like(chosen[:, i], k + 1), chosen[:, i])

    if base is not None:
        rows = rows.masked_fill(~base, _NEG)
    return chosen, torch.stack(lps, 0).sum(0), torch.stack(ents, 0).sum(0), rows


__all__ = ["CtdeNet", "LocalActor", "OpsAttentionNet", "OpsAttentionPolicy", "count_parameters"]
