"""MXFP4 with per-block optimal exponent search.

OCP MXFP4 mandates a deterministic per-block scale:
    E = floor(log2(absmax)) - 2
This is one choice; nothing in the format prevents a different power-of-two
scale (E8M0 still expresses any integer exponent). For each block, we search
E in {default - search_radius, ..., default + search_radius} and pick the
exponent that minimizes the per-block MSE of the dequantized weight against
the FP weight.

This stays inside the MXFP4 format (block-32, E8M0 scale, e2m1 elements):
no extra metadata, same wire footprint as MXFP4. It is a *calibration* choice,
not a format change. Real impls would pre-compute E offline and ship the
same per-block byte as OCP.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from mintengine.quant.methods.mxfp4 import FP4_EMAX, round_to_fp4


@dataclass
class MXFP4Opt:
    """MXFP4 with per-block min-MSE exponent search.

    block_size    : MX spec fixes 32.
    search_radius : try E in [default-r, default+r]. r=0 reproduces OCP MXFP4.
    """

    block_size: int = 32
    search_radius: int = 2

    @property
    def name(self) -> str:
        return f"mxfp4opt-r{self.search_radius}"

    def pack(self, weight: torch.Tensor) -> dict:
        out_f, in_f = weight.shape
        B = self.block_size
        if in_f % B != 0:
            raise ValueError(
                f"MXFP4Opt requires in_features ({in_f}) divisible by block_size ({B})"
            )

        w = weight.detach()
        wb = w.reshape(out_f, in_f // B, B)  # [out, n_blocks, B]

        abs_max = wb.abs().amax(dim=-1, keepdim=True)
        safe_max = abs_max.clamp(min=torch.finfo(w.dtype).tiny)
        E_default = torch.floor(torch.log2(safe_max)) - FP4_EMAX  # [out, n_blocks, 1]

        best_err = None
        best_deq = None
        for offset in range(-self.search_radius, self.search_radius + 1):
            E = E_default + offset
            scale = torch.pow(2.0, E)
            fp4 = round_to_fp4(wb / scale)
            deq = fp4 * scale  # [out, n_blocks, B]
            err = (deq - wb).pow(2).sum(dim=-1, keepdim=True)  # [out, n_blocks, 1]
            if best_err is None:
                best_err = err
                best_deq = deq
            else:
                # Per-block: pick this offset's deq where its err is lower.
                pick = err < best_err  # [out, n_blocks, 1]
                best_deq = torch.where(pick, deq, best_deq)
                best_err = torch.where(pick, err, best_err)

        dequant = best_deq.reshape(out_f, in_f)
        return {"w": dequant.to(weight.dtype)}

    def matmul(self, x: torch.Tensor, packed: dict) -> torch.Tensor:
        return F.linear(x, packed["w"])
