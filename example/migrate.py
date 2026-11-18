"""Widen v1 input layers so a checkpoint can read observation_v2.

New channels start at zero. With those channels zeroed in the observation,
the network matches the v1 checkpoint numerically.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def widen_linear(layer: nn.Linear, new_in: int) -> nn.Linear:
    """Keep old columns and bias; extra input weights are zero."""
    old_in = int(layer.in_features)
    if new_in == old_in:
        return layer

    if new_in < old_in:
        raise ValueError(f"cannot shrink Linear {old_in} -> {new_in}")

    wide = nn.Linear(new_in, layer.out_features, bias=layer.bias is not None)
    with torch.no_grad():
        wide.weight.zero_()
        wide.weight[:, :old_in].copy_(layer.weight)
        if layer.bias is not None:
            wide.bias.copy_(layer.bias)
    return wide


def migrate_observation_v1_to_v2(net: nn.Module, job_dim: int, edge_dim: int) -> list[str]:
    """Expand job and edge embeddings on a dispatcher or CTDE encoder. Returns changed names."""
    changed = []
    encoder = getattr(net, "encoder", None) or getattr(getattr(net, "actor", None), "encoder", None)
    pointer = getattr(net, "pointer", None) or getattr(getattr(net, "actor", None), "pointer", None)
    if encoder is None or pointer is None:
        raise ValueError("checkpoint has no encoder/pointer to widen")

    job_linear = encoder.emb_job[0]
    if not isinstance(job_linear, nn.Linear):
        raise ValueError("emb_job[0] is not Linear")
    if job_linear.in_features != job_dim:
        wide = widen_linear(job_linear, job_dim)
        encoder.emb_job = nn.Sequential(wide, *list(encoder.emb_job.children())[1:])
        changed.append("emb_job")

    if not isinstance(pointer.e, nn.Linear):
        raise ValueError("pointer.e is not Linear")
    if pointer.e.in_features != edge_dim:
        pointer.e = widen_linear(pointer.e, edge_dim)
        changed.append("pointer.e")
    return changed


def overlay_policy_weights(policy: nn.Module, saved: dict, allow_widen: bool) -> list[str]:
    """Copy checkpoint tensors into a live policy.

    Matching shapes copy as-is. A wider Linear in-feature copies the old columns
    and zeros the rest, and only when `allow_widen` is set. Anything else fails
    fast: there is no silent fallback onto random weights.
    """
    current = policy.state_dict()
    widened: list[str] = []
    missing = sorted(key for key in current if key not in saved)
    extra = sorted(key for key in saved if key not in current)
    if extra:
        raise ValueError(f"checkpoint has extra keys {extra[:8]}")
    if missing:
        raise ValueError(f"live net has keys missing from checkpoint {missing[:8]}")

    with torch.no_grad():
        for key, dest in current.items():
            if key not in saved:
                continue
            src = saved[key]
            if not torch.is_tensor(src):
                src = torch.as_tensor(src)
            src = src.to(device=dest.device, dtype=dest.dtype)
            if tuple(src.shape) == tuple(dest.shape):
                dest.copy_(src)
                continue
            if (
                allow_widen
                and dest.ndim == 2
                and src.ndim == 2
                and dest.shape[0] == src.shape[0]
                and dest.shape[1] > src.shape[1]
            ):
                dest.zero_()
                dest[:, : src.shape[1]].copy_(src)
                widened.append(key)
                continue
            raise ValueError(
                f"incompatible tensor {key}: saved {tuple(src.shape)} vs live {tuple(dest.shape)}"
            )
    return widened
