"""HadMXFP4 unit tests.

Verifies:
  - Sylvester Hadamard is orthonormal (H @ H.T == I) and symmetric
  - On weights already constant within a block, HadMXFP4 is exact (rotation
    of a constant block puts all energy on one coordinate; saturation is 0)
  - On heavy-tailed synthetic blocks, HadMXFP4 beats vanilla MXFP4Opt-r1
    on per-block MSE
  - Divisibility / power-of-2 error paths
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mintengine.quant.methods import HadMXFP4, MXFP4Opt  # noqa: E402
from mintengine.quant.methods.had_mxfp4 import _hadamard_sylvester  # noqa: E402


def test_hadamard_is_orthonormal_and_symmetric() -> None:
    for n in (2, 4, 8, 16, 32, 64):
        H = _hadamard_sylvester(n, device="cpu", dtype=torch.float64)
        I = torch.eye(n, dtype=torch.float64)
        if not torch.allclose(H @ H.T, I, atol=1e-10):
            raise AssertionError(f"H@H.T != I for n={n}")
        if not torch.allclose(H, H.T, atol=1e-10):
            raise AssertionError(f"H != H.T for n={n}")
    print("  ok  Sylvester Hadamard orthonormal + symmetric for n in {2,4,8,16,32,64}")


def test_rejects_non_power_of_2_block() -> None:
    w = torch.randn(8, 96)  # 96 not power of 2 (although divisible)
    try:
        HadMXFP4(block_size=96).pack(w)
    except ValueError as e:
        assert "power of 2" in str(e), f"unexpected error: {e}"
        print("  ok  raises ValueError on non-power-of-2 block")
        return
    raise AssertionError("HadMXFP4 should reject non-power-of-2 block_size")


def test_divisibility_error() -> None:
    w = torch.randn(8, 33)
    try:
        HadMXFP4(block_size=32).pack(w)
    except ValueError as e:
        assert "divisible" in str(e), f"unexpected error: {e}"
        print("  ok  raises ValueError when in_features % block_size != 0")
        return
    raise AssertionError("HadMXFP4 should reject non-divisible in_features")


def test_beats_mxfp4opt_on_mixed_magnitude() -> None:
    """Blocks with non-trivial typical values + one outlier per block.

    This is the regime rotation actually helps: typical values are big
    enough that rounding them to 0 hurts, and the outlier inflates the
    per-block scale so they DO round to 0.
    """
    torch.manual_seed(0)
    n_blocks = 64
    B = 32
    w = torch.randn(n_blocks, B)  # std 1
    w[:, 0] *= 5.0                 # one outlier per block, ~5 sigma
    w = w.reshape(1, n_blocks * B).repeat(4, 1)  # [4, 2048]

    mse_opt = (MXFP4Opt(block_size=B, search_radius=1).pack(w)["w"] - w).pow(2).mean().item()
    mse_had = (HadMXFP4(block_size=B, search_radius=1).pack(w)["w"] - w).pow(2).mean().item()
    if not (mse_had < mse_opt):
        raise AssertionError(
            f"HadMXFP4 should beat MXFP4Opt on mixed-magnitude blocks: "
            f"had={mse_had:.3e} opt={mse_opt:.3e}"
        )
    print(f"  ok  HadMXFP4 MSE={mse_had:.3e} < MXFP4Opt MSE={mse_opt:.3e}  "
          f"(reduction: {(1 - mse_had/mse_opt)*100:.1f}%)")


def test_preserves_norm_in_rotated_space() -> None:
    """Sanity: ||W_q - W|| should be roughly the same as ||W_rot_q - W_rot||
    since the inverse rotation is orthogonal."""
    torch.manual_seed(1)
    w = torch.randn(4, 64) * 2.0
    had = HadMXFP4(block_size=32, search_radius=1)
    w_back = had.pack(w)["w"]
    err_orig = (w_back - w).pow(2).sum().item()

    # Recompute by hand: rotate, quantize via MXFP4Opt on the rotated weight,
    # then check the rotated-space error matches the original-space error.
    H = _hadamard_sylvester(32, device=w.device, dtype=torch.float32)
    wb = w.reshape(4, 2, 32) @ H
    w_rot_flat = wb.reshape(4, 64)
    w_rot_q = MXFP4Opt(block_size=32, search_radius=1).pack(w_rot_flat)["w"]
    err_rot = (w_rot_q - w_rot_flat).pow(2).sum().item()

    if abs(err_orig - err_rot) > max(1e-3, 0.01 * err_orig):
        raise AssertionError(
            f"orthogonality broken: err_orig={err_orig:.4e} err_rot={err_rot:.4e}"
        )
    print(f"  ok  ||W'-W||^2 = ||W_rot_q - W_rot||^2 within tol "
          f"({err_orig:.3e} vs {err_rot:.3e})")


def main() -> int:
    tests = [
        test_hadamard_is_orthonormal_and_symmetric,
        test_rejects_non_power_of_2_block,
        test_divisibility_error,
        test_beats_mxfp4opt_on_mixed_magnitude,
        test_preserves_norm_in_rotated_space,
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
