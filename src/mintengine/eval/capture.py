"""Per-layer activation capture via forward hooks on Gemma3Layer."""

from __future__ import annotations

import torch
import torch.nn as nn


@torch.no_grad()
def capture_layer_states(
    decoder: nn.Module,
    input_ids: torch.Tensor,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Run decoder on input_ids, return (per-layer hidden states, final logits).

    Each Gemma3Layer returns (x, residual); the meaningful state at the boundary
    is x + residual (what the next layer's input_layernorm fuses before norm).
    """
    states: list[torch.Tensor] = []
    handles: list[torch.utils.hooks.RemovableHandle] = []
    for layer in decoder.layers:
        def hook(_mod, _inp, out, store=states):
            x, residual = out
            store.append((x + residual).detach().cpu())
        handles.append(layer.register_forward_hook(hook))

    try:
        logits = decoder(input_ids).detach().cpu()
    finally:
        for h in handles:
            h.remove()
    return states, logits
