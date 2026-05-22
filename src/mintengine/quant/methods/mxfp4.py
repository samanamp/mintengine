"""MXFP4 — Open Compute Project Microscaling FP4.

Spec (OCP MX, 2023):
  - Block size: 32 elements along the contracted dim
  - Block scale: E8M0 (signed 8-bit exponent, value = 2^E, no mantissa)
  - Element format: FP4 e2m1 (1 sign + 2 exp + 1 mantissa)
    representable magnitudes = {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}

Fake-quant implementation: produce the dequantized weight, run the fp matmul.
Isolates the format's representational error from any kernel concerns.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


# FP4 e2m1 unsigned magnitudes, ascending.
FP4_MAGS: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# Mid-point thresholds for nearest-rounding (length len(FP4_MAGS) - 1).
FP4_THRESHOLDS: tuple[float, ...] = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
FP4_MAX = FP4_MAGS[-1]
# emax for e2m1: max value 6.0 = 1.5 * 2^2, so emax = 2.
FP4_EMAX = 2


def round_to_fp4(x: torch.Tensor) -> torch.Tensor:
    """Round each element to the nearest signed FP4 e2m1 representable value."""
    thresholds = torch.tensor(FP4_THRESHOLDS, dtype=x.dtype, device=x.device)
    mags = torch.tensor(FP4_MAGS, dtype=x.dtype, device=x.device)
    sign = torch.sign(x)
    mag = x.abs().clamp(max=FP4_MAX)
    idx = torch.bucketize(mag, thresholds, right=True)
    return sign * mags[idx]


@dataclass
class MXFP4:
    """MXFP4 per the OCP microscaling spec.

    block_size : MX spec fixes this at 32; exposed for ablation only.
    """

    block_size: int = 32

    @property
    def name(self) -> str:
        if self.block_size == 32:
            return "mxfp4"
        return f"mxfp4-b{self.block_size}"

    def pack(self, weight: torch.Tensor) -> dict:
        out_f, in_f = weight.shape
        B = self.block_size
        if in_f % B != 0:
            raise ValueError(
                f"MXFP4 requires in_features ({in_f}) divisible by block_size ({B})"
            )

        w = weight.detach()
        wb = w.reshape(out_f, in_f // B, B)  # [out, n_blocks, B]

        # Shared exponent per block: chosen so max(|w|)/2^E lands in the FP4 range.
        # E = floor(log2(max)) - emax_fp4 keeps the largest element near (but not above) FP4_MAX.
        abs_max = wb.abs().amax(dim=-1, keepdim=True)
        # Avoid log2(0); blocks whose max is 0 get scale=1 (any value rounds to 0).
        safe_max = abs_max.clamp(min=torch.finfo(w.dtype).tiny)
        log2_max = torch.log2(safe_max)
        E = torch.floor(log2_max) - FP4_EMAX
        scale = torch.pow(2.0, E)

        x_scaled = wb / scale
        fp4 = round_to_fp4(x_scaled)

        dequant = (fp4 * scale).reshape(out_f, in_f)
        return {"w": dequant.to(weight.dtype)}

    def matmul(self, x: torch.Tensor, packed: dict) -> torch.Tensor:
        return F.linear(x, packed["w"])
