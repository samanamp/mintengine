from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantMethod(Protocol):
    name: str

    def pack(self, weight: torch.Tensor) -> dict:
        ...

    def matmul(self, x: torch.Tensor, packed: dict) -> torch.Tensor:
        ...


@dataclass
class FP:
    """Passthrough — keeps the original weight, used as a correctness baseline.

    Why: apply_quant(model, FP()) must produce bit-identical outputs to the
    untouched model. If it doesn't, the harness itself has a bug.
    """

    name: str = "fp"

    def pack(self, weight: torch.Tensor) -> dict:
        return {"w": weight}

    def matmul(self, x: torch.Tensor, packed: dict) -> torch.Tensor:
        return F.linear(x, packed["w"])


class QLinear(nn.Module):
    def __init__(self, fp_linear: nn.Linear, method: QuantMethod):
        super().__init__()
        self.method = method
        self.in_features = fp_linear.in_features
        self.out_features = fp_linear.out_features
        with torch.no_grad():
            packed = method.pack(fp_linear.weight.data)
        self._packed_keys: list[str] = list(packed.keys())
        for k, v in packed.items():
            if isinstance(v, torch.Tensor):
                self.register_buffer(f"p_{k}", v)
            else:
                setattr(self, f"p_{k}", v)

    def _packed(self) -> dict:
        return {k: getattr(self, f"p_{k}") for k in self._packed_keys}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.method.matmul(x, self._packed())


def apply_quant(module: nn.Module, method: QuantMethod) -> nn.Module:
    """In-place: replace every nn.Linear descendant with QLinear(linear, method)."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, QLinear(child, method))
        else:
            apply_quant(child, method)
    return module
