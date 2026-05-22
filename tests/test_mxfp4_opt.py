"""MXFP4Opt unit tests.

Verifies:
  - r=0 reproduces vanilla MXFP4 exactly
  - r>0 never produces a worse per-block MSE than r=0
  - Divisibility error path
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mintengine.quant.methods import MXFP4, MXFP4Opt  # noqa: E402


def _per_block_mse(deq: torch.Tensor, w: torch.Tensor, B: int = 32) -> torch.Tensor:
    out_f, in_f = w.shape
    return (deq - w).pow(2).reshape(out_f, in_f // B, B).mean(dim=-1)


def test_r0_matches_vanilla_mxfp4() -> None:
    torch.manual_seed(0)
    w = torch.randn(8, 64) * 3.0
    vanilla = MXFP4(block_size=32).pack(w)["w"]
    opt0 = MXFP4Opt(block_size=32, search_radius=0).pack(w)["w"]
    if not torch.equal(vanilla, opt0):
        diff = (vanilla - opt0).abs().max().item()
        raise AssertionError(f"r=0 != vanilla MXFP4, max diff {diff:.3e}")
    print("  ok  r=0 reproduces vanilla MXFP4")


def test_r_positive_never_worse() -> None:
    torch.manual_seed(1)
    w = torch.randn(8, 128) * 4.0
    vanilla = MXFP4(block_size=32).pack(w)["w"]
    vmse = _per_block_mse(vanilla, w)
    for r in (1, 2, 3):
        opt = MXFP4Opt(block_size=32, search_radius=r).pack(w)["w"]
        omse = _per_block_mse(opt, w)
        if not (omse <= vmse + 1e-6).all():
            excess = (omse - vmse).max().item()
            raise AssertionError(
                f"r={r}: per-block MSE worse than vanilla by {excess:.3e}"
            )
        improvement = (vmse - omse).sum().item()
        print(f"  ok  r={r} never worse than vanilla  (total MSE reduction: {improvement:.3e})")


def test_divisibility_error() -> None:
    w = torch.randn(8, 33)
    try:
        MXFP4Opt(block_size=32).pack(w)
    except ValueError as e:
        assert "divisible" in str(e), f"unexpected error: {e}"
        print("  ok  raises ValueError when in_features % block_size != 0")
        return
    raise AssertionError("MXFP4Opt.pack should have raised on non-divisible shape")


def main() -> int:
    tests = [
        test_r0_matches_vanilla_mxfp4,
        test_r_positive_never_worse,
        test_divisibility_error,
    ]
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
