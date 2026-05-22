"""Activation-aware Weight Quantization (AWQ), fake-quant edition.

Sketch:
  for each nn.Linear with calib activations X (shape [..., in_features]):
    act_mag = mean(|X|, over all dims except last)          # [in]
    for alpha in alpha_grid:
        s = act_mag ** alpha                                 # [in]
        s = s / s.mean()                                     # normalize
        s = s.clamp(min=1e-4)                                # safety
        W_scaled = W * s                                     # [out, in] * [in]
        W_q = RTN(bits, group_size).pack(W_scaled)['w']      # dequantized weight
        # output reconstruction loss on the calib batch:
        Y_ref = F.linear(X, W)                               # FP target
        Y_awq = F.linear(X / s, W_q)                         # scaled+quant path
        loss = mse(Y_ref, Y_awq)
    pick alpha with lowest loss
    replace Linear with AWQLinear(s_best, W_q_best)

At inference: Y = F.linear(X / s, W_q). Mathematically equivalent to running
W' = W * s through quantization in input-channel-scaled space, then undoing
the scale on the input side. In production this scale fuses into the
preceding norm; here we just multiply.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from mintengine.quant.methods.rtn import RTN


@dataclass
class AWQConfig:
    bits: int = 4
    group_size: int | None = 128
    alpha_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

    @property
    def name(self) -> str:
        g = "row" if self.group_size is None else f"g{self.group_size}"
        return f"awq{self.bits}-{g}"


class AWQLinear(nn.Module):
    """Wraps a pre-quantized scaled weight + the per-input-channel inverse scale."""

    def __init__(self, w_q: torch.Tensor, scale: torch.Tensor) -> None:
        super().__init__()
        # w_q is the dequantized (fake-quant) version of W * scale, shape [out, in]
        self.register_buffer("w_q", w_q)
        # scale is per-input-channel, shape [in_features]; store inverse to mul not div
        self.register_buffer("inv_scale", 1.0 / scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x * self.inv_scale, self.w_q)


def _collect_activation_magnitudes(
    decoder: nn.Module, calib_ids: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Hook every nn.Linear; record mean(|X|) per input channel over the calib batch.

    Returns a dict mapping module path -> tensor of shape [in_features].
    """
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    handles = []

    def make_hook(path: str):
        def hook(_mod, inputs, _out):
            x = inputs[0].detach()
            # Reduce all dims except the last (in_features)
            mag = x.abs().float().reshape(-1, x.shape[-1])  # [N, in]
            s = mag.sum(dim=0)  # [in]
            n = mag.shape[0]
            if path in sums:
                sums[path] += s
                counts[path] += n
            else:
                sums[path] = s
                counts[path] = n
        return hook

    for name, mod in decoder.named_modules():
        if isinstance(mod, nn.Linear):
            handles.append(mod.register_forward_hook(make_hook(name)))

    try:
        with torch.no_grad():
            _ = decoder(calib_ids)
    finally:
        for h in handles:
            h.remove()

    return {path: (sums[path] / counts[path]).to(torch.float32) for path in sums}


def _best_alpha(
    weight: torch.Tensor,
    calib_x: torch.Tensor,
    act_mag: torch.Tensor,
    cfg: AWQConfig,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Grid-search alpha; return (alpha*, w_q*, scale*)."""
    rtn = RTN(bits=cfg.bits, group_size=cfg.group_size)
    # FP reference output on calib batch
    with torch.no_grad():
        y_ref = F.linear(calib_x, weight)

    best = (None, None, None, float("inf"))
    for alpha in cfg.alpha_grid:
        if alpha == 0.0:
            s = torch.ones_like(act_mag)
        else:
            s = act_mag.pow(alpha)
            s = s / s.mean()
            s = s.clamp(min=1e-4)
        w_scaled = weight * s
        w_q = rtn.pack(w_scaled)["w"]
        with torch.no_grad():
            y_q = F.linear(calib_x * (1.0 / s), w_q)
            loss = F.mse_loss(y_q.float(), y_ref.float()).item()
        if loss < best[3]:
            best = (alpha, w_q, s, loss)
    alpha_star, w_q_star, s_star, _ = best
    return float(alpha_star), w_q_star, s_star


def _collect_layer_inputs(
    decoder: nn.Module, calib_ids: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Capture the *input tensor* into each nn.Linear over the calib batch.

    Memory-heavy (stores the full activation for every Linear) but simple,
    and fine at calibration-batch scale (a couple of short sequences).
    """
    inputs_cache: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(path: str):
        def hook(_mod, inputs, _out):
            inputs_cache[path] = inputs[0].detach().cpu()
        return hook

    for name, mod in decoder.named_modules():
        if isinstance(mod, nn.Linear):
            handles.append(mod.register_forward_hook(make_hook(name)))

    try:
        with torch.no_grad():
            _ = decoder(calib_ids)
    finally:
        for h in handles:
            h.remove()
    return inputs_cache


def apply_awq(
    decoder: nn.Module, calib_ids: torch.Tensor, cfg: AWQConfig | None = None
) -> dict[str, float]:
    """In-place: replace every nn.Linear in decoder with an AWQLinear.

    Returns a dict mapping module path -> chosen alpha (for diagnostics).
    """
    cfg = cfg or AWQConfig()
    act_mag = _collect_activation_magnitudes(decoder, calib_ids)
    calib_x = _collect_layer_inputs(decoder, calib_ids)

    chosen: dict[str, float] = {}

    # Walk the tree, replacing in parents
    def recurse(module: nn.Module, prefix: str = "") -> None:
        for name, child in list(module.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear):
                weight = child.weight.detach()
                if full not in act_mag or full not in calib_x:
                    # Layer never ran during calibration — fall back to RTN equivalent
                    s = torch.ones(weight.shape[1], device=weight.device,
                                    dtype=torch.float32)
                    w_q = RTN(bits=cfg.bits, group_size=cfg.group_size).pack(weight)["w"]
                    chosen[full] = 0.0
                else:
                    x = calib_x[full].to(weight.device).to(weight.dtype)
                    am = act_mag[full].to(weight.device)
                    alpha, w_q, s = _best_alpha(weight, x, am, cfg)
                    chosen[full] = alpha
                setattr(module, name, AWQLinear(w_q.to(weight.device),
                                                  s.to(weight.device)))
            else:
                recurse(child, full)

    recurse(decoder)
    return chosen
