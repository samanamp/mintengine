"""Quantization-error metrics.

layer_mse: cheap per-layer proxy, optimized in the inner loop.
output_kl: end-to-end arbiter, the only number that tracks downstream quality.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def layer_mse(
    states_ref: list[torch.Tensor],
    states_q: list[torch.Tensor],
) -> list[float]:
    if len(states_ref) != len(states_q):
        raise ValueError(
            f"layer count mismatch: ref={len(states_ref)} q={len(states_q)}"
        )
    return [
        F.mse_loss(a.float(), b.float()).item()
        for a, b in zip(states_ref, states_q)
    ]


def output_kl(logits_ref: torch.Tensor, logits_q: torch.Tensor) -> float:
    """KL(P_ref || P_q) averaged over batch and sequence positions."""
    log_p_ref = F.log_softmax(logits_ref.float(), dim=-1)
    log_p_q = F.log_softmax(logits_q.float(), dim=-1)
    p_ref = log_p_ref.exp()
    return (p_ref * (log_p_ref - log_p_q)).sum(dim=-1).mean().item()
