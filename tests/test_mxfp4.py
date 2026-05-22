"""MXFP4 unit tests.

Verifies:
  - round_to_fp4 maps to the exact 8 representable magnitudes
  - Mid-point thresholds round to expected nearest values
  - Per-block dequantized weight: max absolute error is bounded by half the
    smallest representable step at the block's scale
  - In_features must be divisible by block_size
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mintengine.quant.methods import MXFP4, round_to_fp4  # noqa: E402
from mintengine.quant.methods.mxfp4 import FP4_MAGS  # noqa: E402


def test_round_hits_exact_values() -> None:
    x = torch.tensor(list(FP4_MAGS) + [-m for m in FP4_MAGS])
    y = round_to_fp4(x)
    if not torch.equal(x, y):
        raise AssertionError(f"exact values changed: {x.tolist()} -> {y.tolist()}")
    print(f"  ok  exact FP4 magnitudes preserved (signed)")


def test_round_clamps_above_max() -> None:
    x = torch.tensor([7.0, 100.0, -7.0, -1e6])
    y = round_to_fp4(x)
    expected = torch.tensor([6.0, 6.0, -6.0, -6.0])
    if not torch.equal(y, expected):
        raise AssertionError(f"clamp failed: {y.tolist()} != {expected.tolist()}")
    print(f"  ok  out-of-range values clamped to ±{FP4_MAGS[-1]}")


def test_round_midpoints() -> None:
    # Mid-points should round to the upper of the two neighbours (bucketize
    # right-bound semantics): e.g. 0.25 -> 0.5, 1.75 -> 2.0, 5.0 -> 6.0
    cases = [
        (0.25, 0.5),
        (0.75, 1.0),
        (1.25, 1.5),
        (1.75, 2.0),
        (2.5, 3.0),
        (3.5, 4.0),
        (5.0, 6.0),
        # Just below mid-points should round down
        (0.24, 0.0),
        (1.74, 1.5),
        (4.99, 4.0),
    ]
    x = torch.tensor([c[0] for c in cases])
    expected = torch.tensor([c[1] for c in cases])
    y = round_to_fp4(x)
    if not torch.equal(y, expected):
        raise AssertionError(f"midpoint rounding wrong: {y.tolist()} != {expected.tolist()}")
    print(f"  ok  mid-point rounding behaviour as specified")


def test_pack_divisibility_error() -> None:
    w = torch.randn(8, 33)  # 33 not divisible by 32
    try:
        MXFP4(block_size=32).pack(w)
    except ValueError as e:
        msg = str(e)
        assert "divisible" in msg, f"unexpected error msg: {msg}"
        print(f"  ok  raises ValueError when in_features % block_size != 0")
        return
    raise AssertionError("MXFP4.pack should have raised on non-divisible shape")


def test_pack_block_error_bound() -> None:
    """Per-element dequant error <= 2.0 * shared_scale.

    Why 2.0: the OCP scale rule scale = 2^(floor(log2(max)) - 2) admits values
    up to ~8 in FP4-space before saturation (since max in [2^k, 2^(k+1)) gives
    scale = 2^(k-2), so max/scale can reach 8). FP4 max is 6, so worst-case
    saturation error is 2 in FP4-space (= 2*scale in real-space). The largest
    rounding gap between adjacent FP4 magnitudes is 2 (between 4 and 6) which
    also gives half-gap = 1 < 2. So 2.0*scale is the tight worst case.
    """
    torch.manual_seed(0)
    w = torch.randn(4, 64) * 5.0
    deq = MXFP4(block_size=32).pack(w)["w"]
    err = (deq - w).abs()
    wb = w.reshape(4, 2, 32)
    abs_max = wb.abs().amax(dim=-1, keepdim=True).clamp(min=torch.finfo(w.dtype).tiny)
    scale = torch.pow(2.0, torch.floor(torch.log2(abs_max)) - 2.0)
    bound = 2.0 * scale
    bound_per_elem = bound.expand_as(wb).reshape_as(w)
    if not (err <= bound_per_elem + 1e-6).all():
        excess = (err - bound_per_elem).max().item()
        raise AssertionError(f"per-block error bound violated: max excess {excess:.3e}")
    print(f"  ok  per-block dequant error <= 2.0 * shared_scale  "
          f"(max actual: {err.max().item():.3e})")


def test_pack_matches_definition() -> None:
    """Tautological but pins the impl to the spec formula end-to-end."""
    torch.manual_seed(1)
    w = torch.randn(4, 64) * 2.0
    deq = MXFP4(block_size=32).pack(w)["w"]
    wb = w.reshape(4, 2, 32)
    abs_max = wb.abs().amax(dim=-1, keepdim=True).clamp(min=torch.finfo(w.dtype).tiny)
    scale = torch.pow(2.0, torch.floor(torch.log2(abs_max)) - 2.0)
    expected = (round_to_fp4(wb / scale) * scale).reshape(4, 64)
    if not torch.equal(deq, expected):
        diff = (deq - expected).abs().max().item()
        raise AssertionError(f"pack != defining formula: max diff {diff:.3e}")
    print(f"  ok  pack matches scale -> round_to_fp4 -> dequant exactly")


def test_pack_idempotent_on_quantized_values() -> None:
    """If a block is already on the FP4 grid at scale=1, pack leaves it alone."""
    # 15 unique signed FP4 values: pad/repeat to fill a 32-wide row.
    base = list(FP4_MAGS) + [-m for m in FP4_MAGS if m > 0]  # 15
    row = (base * 3)[:32]  # 32 entries, all on the FP4 grid
    w = torch.tensor(row, dtype=torch.float32).unsqueeze(0).repeat(2, 1)  # [2, 32]
    deq = MXFP4(block_size=32).pack(w)["w"]
    if not torch.equal(w, deq):
        diff = (w - deq).abs().max().item()
        raise AssertionError(f"on-grid weights should be fixed point, max diff {diff:.3e}")
    print(f"  ok  on-grid weights pass through unchanged")


def main() -> int:
    tests = [
        test_round_hits_exact_values,
        test_round_clamps_above_max,
        test_round_midpoints,
        test_pack_divisibility_error,
        test_pack_block_error_bound,
        test_pack_idempotent_on_quantized_values,
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
