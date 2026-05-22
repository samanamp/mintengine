"""Input generators for quantization eval — no external calibration data.

random_tokens: ultra-fast, zero on-manifold guarantee, good for inner-loop ranking.
self_generated_rollouts: model talks to itself from BOS, hits the real activation
manifold (including outlier channels) without committing to any data domain.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def random_tokens(
    vocab_size: int,
    batch: int = 4,
    length: int = 128,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.randint(0, vocab_size, (batch, length), generator=g)
    return ids.to(device)


@torch.no_grad()
def self_generated_rollouts(
    decoder: nn.Module,
    bos_id: int,
    num_seqs: int = 4,
    length: int = 128,
    temperature: float = 1.0,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Sample sequences from the model itself starting from a single BOS token.

    O(L^2) without a KV cache — keep `length` modest until that lands.
    """
    g_cpu = torch.Generator(device="cpu").manual_seed(seed)
    seqs = torch.full((num_seqs, 1), bos_id, dtype=torch.long, device=device)
    for _ in range(length - 1):
        logits = decoder(seqs)[:, -1, :].float() / max(temperature, 1e-8)
        probs = torch.softmax(logits, dim=-1).cpu()
        next_tok = torch.multinomial(probs, num_samples=1, generator=g_cpu).to(device)
        seqs = torch.cat([seqs, next_tok], dim=1)
    return seqs
