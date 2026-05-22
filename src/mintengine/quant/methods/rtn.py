from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class RTN:
    """Symmetric round-to-nearest, fake-quantized.

    bits        : quantization bit-width (signed; e.g. 4 -> [-8, 7])
    group_size  : None for per-output-row; an int G to share a scale
                  across each contiguous group of G input dims (per-row
                  splits into in_features/G groups). Smaller G = more
                  scales = less error but more overhead.

    Stores the dequantized weight so the fp matmul path is reused;
    isolates the *scheme's* error from any kernel concerns.
    """

    bits: int = 4
    group_size: int | None = None

    @property
    def name(self) -> str:
        g = "row" if self.group_size is None else f"g{self.group_size}"
        return f"rtn{self.bits}-{g}"

    def pack(self, weight: torch.Tensor) -> dict:
        qmax = 2 ** (self.bits - 1) - 1
        qmin = -(2 ** (self.bits - 1))
        w = weight.detach()
        out_f, in_f = w.shape
        G = self.group_size

        if G is None or in_f % G != 0:
            scale = w.abs().amax(dim=1, keepdim=True) / qmax
            scale = scale.clamp(min=torch.finfo(weight.dtype).tiny)
            q = torch.round(w / scale).clamp(qmin, qmax)
            deq = q * scale
        else:
            wg = w.reshape(out_f, in_f // G, G)
            scale = wg.abs().amax(dim=2, keepdim=True) / qmax
            scale = scale.clamp(min=torch.finfo(weight.dtype).tiny)
            q = torch.round(wg / scale).clamp(qmin, qmax)
            deq = (q * scale).reshape(out_f, in_f)

        return {"w": deq.to(weight.dtype)}

    def matmul(self, x: torch.Tensor, packed: dict) -> torch.Tensor:
        return F.linear(x, packed["w"])
