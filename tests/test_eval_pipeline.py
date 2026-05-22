"""Eval pipeline regression: capture + metrics on a Gemma3-shaped mock decoder.

Verifies:
 - capture_layer_states returns one state per layer + final logits
 - FP-vs-FP run produces zero layer MSE and zero output KL (harness sanity)
 - RTN run produces strictly positive layer MSE (the signal isn't dead)

Runnable as a script: uv run python tests/test_eval_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mintengine.eval import (  # noqa: E402
    capture_layer_states,
    layer_mse,
    output_kl,
    random_tokens,
)
from mintengine.quant import FP, apply_quant  # noqa: E402
from mintengine.quant.methods import RTN  # noqa: E402


# --- Gemma3-shaped mock --------------------------------------------------------


class MockLayer(nn.Module):
    """Same (x, residual) contract as Gemma3Layer."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(dim, dim, bias=False)

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = x
            x = self.lin(x)
        else:
            x = self.lin(x + residual)
            residual = x
        return x, residual


class MockDecoder(nn.Module):
    def __init__(self, vocab_size: int, dim: int, num_layers: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([MockLayer(dim) for _ in range(num_layers)])
        self.out = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.embed(ids)
        residual: torch.Tensor | None = None
        for layer in self.layers:
            h, residual = layer(h, residual)
        return self.out(h + residual)


def _build(seed: int = 0) -> tuple[MockDecoder, int]:
    torch.manual_seed(seed)
    vocab, dim, layers = 257, 64, 4
    return MockDecoder(vocab, dim, layers), vocab


# --- Tests --------------------------------------------------------------------


def test_capture_shapes() -> None:
    model, vocab = _build()
    ids = random_tokens(vocab, batch=2, length=8)
    states, logits = capture_layer_states(model, ids)
    assert len(states) == len(model.layers), f"got {len(states)} states"
    assert logits.shape == (2, 8, vocab), f"got {logits.shape}"
    print(f"  ok  {len(states)} states captured, logits {tuple(logits.shape)}")


def test_fp_run_is_zero_signal() -> None:
    model, vocab = _build()
    ids = random_tokens(vocab, batch=2, length=8)

    states_a, logits_a = capture_layer_states(model, ids)
    apply_quant(model, FP())
    states_b, logits_b = capture_layer_states(model, ids)

    mses = layer_mse(states_a, states_b)
    kl = output_kl(logits_a, logits_b)
    if any(m != 0.0 for m in mses):
        raise AssertionError(f"FP-vs-FP layer MSE not zero: {mses}")
    if kl != 0.0:
        raise AssertionError(f"FP-vs-FP output KL not zero: {kl}")
    print(f"  ok  FP-vs-FP: layer_mse all zero, kl={kl}")


def test_rtn_run_has_signal() -> None:
    model, vocab = _build()
    ids = random_tokens(vocab, batch=2, length=8)

    states_fp, logits_fp = capture_layer_states(model, ids)
    apply_quant(model, RTN(bits=2))  # aggressive: guarantees nonzero error
    states_q, logits_q = capture_layer_states(model, ids)

    mses = layer_mse(states_fp, states_q)
    kl = output_kl(logits_fp, logits_q)
    if not all(m > 0.0 for m in mses):
        raise AssertionError(f"RTN should perturb every layer, got {mses}")
    if not (kl > 0.0):
        raise AssertionError(f"RTN should perturb output KL, got {kl}")
    print(f"  ok  RTN@2bit: per-layer mse={[f'{m:.2e}' for m in mses]}, kl={kl:.3e}")


def main() -> int:
    tests = [test_capture_shapes, test_fp_run_is_zero_signal, test_rtn_run_has_signal]
    failed = 0
    for t in tests:
        print(f"\n[{t.__name__}]")
        try:
            t()
        except AssertionError as e:
            print(f"  FAIL  {e}")
            failed += 1
    print()
    if failed:
        print(f"{failed}/{len(tests)} failed")
        return 1
    print(f"all {len(tests)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
