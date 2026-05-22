"""FP-passthrough regression: apply_quant(model, FP()) must be a no-op.

If this ever fails, the quant scaffold itself is wrong — every subsequent
quant-method measurement would be contaminated by the harness.

Runnable as a script:
    uv run python tests/test_quant_fp.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mintengine.models.gemma3 import Gemma3MLP  # noqa: E402
from mintengine.models.gemma3_weights import MLPWeights  # noqa: E402
from mintengine.quant import FP, QLinear, apply_quant, restore_fp, snapshot_linears  # noqa: E402
from mintengine.quant.methods import RTN  # noqa: E402


def _assert_identical(a: torch.Tensor, b: torch.Tensor, label: str) -> None:
    if not torch.equal(a, b):
        diff = (a - b).abs().max().item()
        raise AssertionError(f"{label}: outputs differ, max |Δ| = {diff:.3e}")
    print(f"  ok  {label}: bit-identical")


def test_fp_single_linear() -> None:
    torch.manual_seed(0)
    lin = nn.Linear(64, 32, bias=False)
    x = torch.randn(2, 8, 64)
    y_ref = lin(x)

    q = QLinear(lin, FP())
    y_q = q(x)
    _assert_identical(y_ref, y_q, "single nn.Linear")


def test_fp_sequential_walked() -> None:
    torch.manual_seed(1)
    model = nn.Sequential(
        nn.Linear(48, 96, bias=False),
        nn.GELU(approximate="tanh"),
        nn.Linear(96, 48, bias=False),
    )
    x = torch.randn(4, 16, 48)
    y_ref = model(x)

    apply_quant(model, FP())
    # confirm walker actually replaced both Linears
    replaced = sum(isinstance(m, QLinear) for m in model.modules())
    assert replaced == 2, f"expected 2 QLinear, found {replaced}"

    y_q = model(x)
    _assert_identical(y_ref, y_q, "nn.Sequential (2 Linears)")


def test_mlp_refactor_matches_old_formulation() -> None:
    """The MLP refactor (raw matmul -> nn.Linear) must be a numerical no-op.

    Recomputes the pre-refactor formula inline and compares to the current
    Gemma3MLP forward. Protects the user's HF-parity guarantee from drift.
    """
    torch.manual_seed(7)
    hidden, intermediate = 96, 192
    gate = torch.randn(intermediate, hidden)
    up = torch.randn(intermediate, hidden)
    down = torch.randn(hidden, intermediate)

    w = MLPWeights()
    w.gate_proj, w.up_proj, w.down_proj = gate, up, down
    mlp = Gemma3MLP(w)
    x = torch.randn(1, 7, hidden)
    y_new = mlp(x)

    gate_up_w = torch.cat((gate, up), dim=0).T  # old layout: [hidden, 2*inter]
    gu = torch.matmul(x, gate_up_w)
    g, u = torch.split(gu, intermediate, dim=-1)
    g = nn.GELU(approximate="tanh")(g)
    y_old = torch.matmul(g * u, down.T)

    _assert_identical(y_new, y_old, "MLP refactor matches pre-refactor matmul")


def test_fp_gemma3_mlp() -> None:
    """End-to-end on the real MLP block — exercises gate/up split + down_proj."""
    torch.manual_seed(2)
    hidden, intermediate = 128, 256
    w = MLPWeights()
    w.gate_proj = torch.randn(intermediate, hidden)
    w.up_proj = torch.randn(intermediate, hidden)
    w.down_proj = torch.randn(hidden, intermediate)

    mlp = Gemma3MLP(w)
    x = torch.randn(1, 5, hidden)
    y_ref = mlp(x)

    apply_quant(mlp, FP())
    replaced = sum(isinstance(m, QLinear) for m in mlp.modules())
    assert replaced == 2, f"expected 2 QLinear in MLP, found {replaced}"

    y_q = mlp(x)
    _assert_identical(y_ref, y_q, "Gemma3MLP (gate_up + down)")


def test_where_filter_skips() -> None:
    """`where` should restrict apply_quant to a subset of Linears."""
    torch.manual_seed(3)
    model = nn.Sequential(
        nn.Linear(16, 32, bias=False),
        nn.Linear(32, 16, bias=False),
    )
    # only quantize the second linear (path "1")
    apply_quant(model, RTN(bits=2), where=lambda path, _: path == "1")
    types = [type(m).__name__ for m in model]
    assert types[0] == "Linear", f"first layer should be untouched, got {types[0]}"
    assert types[1] == "QLinear", f"second layer should be QLinear, got {types[1]}"
    print(f"  ok  where-filter: types = {types}")


def test_snapshot_restore_roundtrip() -> None:
    """snapshot -> apply_quant(RTN2) -> restore_fp must reproduce FP outputs."""
    torch.manual_seed(4)
    model = nn.Sequential(
        nn.Linear(24, 48, bias=False),
        nn.GELU(),
        nn.Linear(48, 24, bias=False),
    )
    x = torch.randn(2, 6, 24)
    y_ref = model(x)

    snap = snapshot_linears(model)
    apply_quant(model, RTN(bits=2))
    y_q = model(x)
    if torch.equal(y_ref, y_q):
        raise AssertionError("RTN@2bit should perturb output but didn't")

    restore_fp(model, snap)
    y_restored = model(x)
    _assert_identical(y_ref, y_restored, "snapshot/restore roundtrip")
    types = [type(m).__name__ for m in model]
    assert "QLinear" not in types, f"restore left QLinears behind: {types}"


def main() -> int:
    tests = [
        test_fp_single_linear,
        test_fp_sequential_walked,
        test_mlp_refactor_matches_old_formulation,
        test_fp_gemma3_mlp,
        test_where_filter_skips,
        test_snapshot_restore_roundtrip,
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
