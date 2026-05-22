"""Hadamard-rotated MXFP4 with per-block optimal exponent search.

Why a rotation helps:
  Per-block quantization scales to absmax. A block dominated by a single
  outlier wastes the FP4 grid on that one value; the other 31 elements
  share a coarse scale. Mixing the 32 block values via an orthogonal
  rotation (here: the size-32 Sylvester Hadamard) produces a block whose
  values are linear combinations of the originals — central-limit-like,
  smaller dynamic range, fewer extreme outliers. MXFP4 quantizes that
  better.

  We rotate in the contracted (in_features) direction *within each block*,
  so the rotation is block-diagonal and undoes cleanly post-quant. Since
  the rotation is orthogonal, applying it then undoing it on the
  dequantized weight is mathematically exact (no extra runtime cost):
  rotate -> MXFP4 -> rotate-back is a self-contained calibration step,
  producing a normal weight matrix W' that is dropped in as-is.

  Note: error norm is preserved by an orthogonal map, so this *only*
  helps because the rotated weights round better per-block — not because
  the rotation hides error. If rotated weights round identically to
  originals (e.g. they're already uniform), no improvement; if they're
  more uniform (typical), big improvement.

Format-compatibility caveat:
  Real production MXFP4 ships the per-block byte (E8M0) and the e2m1
  elements. A Hadamard rotation pre-applied to W and not undone would
  change the runtime semantics (would need to rotate the input too).
  Here we *undo* the rotation in the dequantized weight, so W' is
  drop-in — this is purely a *weight preprocessing* trick, not a format
  extension. Trade-off: the savings vs format-native QuaRot-style schemes
  is that we can't fuse the rotation into the preceding norm; we pay the
  rotation cost only at calibration, not at runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from mintengine.quant.methods.mxfp4 import FP4_EMAX, round_to_fp4


def _hadamard_sylvester(n: int, device, dtype) -> torch.Tensor:
    """Orthonormal Sylvester Hadamard of size n (n must be a power of 2).

    H @ H.T = I (and H is symmetric).
    """
    if n & (n - 1) != 0 or n == 0:
        raise ValueError(f"Hadamard size must be a positive power of 2, got {n}")
    H = torch.ones(1, 1, dtype=dtype, device=device)
    while H.shape[0] < n:
        top = torch.cat([H, H], dim=1)
        bot = torch.cat([H, -H], dim=1)
        H = torch.cat([top, bot], dim=0)
    return H / math.sqrt(n)


@dataclass
class HadMXFP4:
    """Hadamard-rotated MXFP4 with min-MSE per-block exponent search.

    block_size    : MX spec fixes 32 (must be power of 2 here too).
    search_radius : exponent search half-width (0 = OCP rule, 1 captures the win).
    """

    block_size: int = 32
    search_radius: int = 1

    @property
    def name(self) -> str:
        return f"hadmxfp4-r{self.search_radius}"

    def pack(self, weight: torch.Tensor) -> dict:
        out_f, in_f = weight.shape
        B = self.block_size
        if in_f % B != 0:
            raise ValueError(
                f"HadMXFP4 requires in_features ({in_f}) divisible by block_size ({B})"
            )

        w = weight.detach()
        # Use float32 for the math; cast back at the end.
        work_dtype = torch.float32
        w32 = w.to(work_dtype)
        H = _hadamard_sylvester(B, device=w.device, dtype=work_dtype)  # [B, B]

        wb = w32.reshape(out_f, in_f // B, B)        # [out, n_blk, B]
        wb_rot = wb @ H                               # rotate last dim (H symmetric)

        abs_max = wb_rot.abs().amax(dim=-1, keepdim=True)
        safe_max = abs_max.clamp(min=torch.finfo(work_dtype).tiny)
        E_default = torch.floor(torch.log2(safe_max)) - FP4_EMAX  # [out, n_blk, 1]

        best_err = None
        best_deq_rot = None
        for offset in range(-self.search_radius, self.search_radius + 1):
            E = E_default + offset
            scale = torch.pow(2.0, E)
            fp4 = round_to_fp4(wb_rot / scale)
            deq_rot = fp4 * scale
            err = (deq_rot - wb_rot).pow(2).sum(dim=-1, keepdim=True)
            if best_err is None:
                best_err = err
                best_deq_rot = deq_rot
            else:
                pick = err < best_err
                best_deq_rot = torch.where(pick, deq_rot, best_deq_rot)
                best_err = torch.where(pick, err, best_err)

        # Undo rotation (H is symmetric & orthogonal, so H == H.T == H^{-1}).
        wb_back = best_deq_rot @ H                    # [out, n_blk, B]
        dequant = wb_back.reshape(out_f, in_f)
        return {"w": dequant.to(weight.dtype)}

    def matmul(self, x: torch.Tensor, packed: dict) -> torch.Tensor:
        return F.linear(x, packed["w"])
