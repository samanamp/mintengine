from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class RTN:
    """Per-output-row symmetric round-to-nearest, fake-quantized.

    Stores the dequantized weight so the fp matmul path is reused; this
    isolates the *scheme's* error from any kernel concerns.
    """

    bits: int = 4
    name: str = "rtn"

    def pack(self, weight: torch.Tensor) -> dict:
        qmax = 2 ** (self.bits - 1) - 1
        qmin = -(2 ** (self.bits - 1))
        scale = weight.detach().abs().amax(dim=1, keepdim=True) / qmax
        scale = scale.clamp(min=torch.finfo(weight.dtype).tiny)
        q = torch.round(weight / scale).clamp(qmin, qmax)
        return {"w": (q * scale).to(weight.dtype)}

    def matmul(self, x: torch.Tensor, packed: dict) -> torch.Tensor:
        return F.linear(x, packed["w"])
