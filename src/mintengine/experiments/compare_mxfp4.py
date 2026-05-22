"""MXFP4 vs RTN at matched footprint, scored on trustworthy inputs.

Why this comparison:
  - MXFP4 is block-32, ~4.25 bits/weight (4 bits element + E8M0 shared scale per
    32 elements ≈ 0.25 bit amortized). The closest INT4 comparators are
    rtn4_g32 (same block) and rtn4_g8 (the RTN sweet spot from tighten_finding).
  - We learned that random tokens scramble rankings; score on self_s2 + real_text
    (the two trustworthy signals from tighten_finding).

Run:  uv run python -m mintengine.experiments.compare_mxfp4
"""

from __future__ import annotations

import contextlib
import io
import json
import time
from pathlib import Path

import torch

from mintengine.eval import (
    capture_layer_states,
    layer_mse,
    output_kl,
    self_generated_rollouts,
)
from mintengine.models.gemma3 import Gemma3
from mintengine.quant import apply_quant, restore_fp, snapshot_linears
from mintengine.quant.methods import MXFP4, MXFP4Opt, RTN


REPORT_DIR = Path(__file__).resolve().parents[3] / "reports" / "quant"

CONFIGS = [
    ("rtn4_row",  RTN(bits=4)),
    ("rtn4_g128", RTN(bits=4, group_size=128)),
    ("rtn4_g32",  RTN(bits=4, group_size=32)),
    ("rtn4_g8",   RTN(bits=4, group_size=8)),
    ("mxfp4",     MXFP4(block_size=32)),
    ("mxfp4opt1", MXFP4Opt(block_size=32, search_radius=1)),
    ("mxfp4opt2", MXFP4Opt(block_size=32, search_radius=2)),
]

REAL_TEXT = (
    "The development of large language models has accelerated rapidly over "
    "the last several years, with new architectures and training techniques "
    "emerging from research labs around the world. One particularly active "
    "area of investigation is post-training quantization, where the weights "
    "of a trained model are reduced to lower precision in order to fit on "
    "smaller hardware or to run with lower latency. A central challenge in "
    "this area is preserving the statistical behavior of activations, "
    "especially at the small number of outlier channels that tend to carry "
    "disproportionate information through the network."
)


@contextlib.contextmanager
def _silent():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def _fp_capture(decoder, ids):
    with _silent():
        states, logits = capture_layer_states(decoder, ids)
    return states, logits


def _score(decoder, ids, fp_states, fp_logits, snap, label):
    print(f"\n--- {label}  shape={tuple(ids.shape)} ---")
    out = []
    for name, method in CONFIGS:
        apply_quant(decoder, method)
        try:
            with _silent():
                q_states, q_logits = capture_layer_states(decoder, ids)
        finally:
            restore_fp(decoder, snap)
        kl = output_kl(fp_logits, q_logits)
        mses = layer_mse(fp_states, q_states)
        out.append({
            "config": name,
            "output_kl": kl,
            "max_mse": max(mses),
            "worst_layer": mses.index(max(mses)),
            "layer_mse": mses,
        })
        print(f"  {name:12s} KL={kl:.4e}  max_mse={max(mses):.2e}@L{mses.index(max(mses))}")
    return out


def _rank(results):
    return [r["config"] for r in sorted(results, key=lambda r: r["output_kl"])]


def main() -> int:
    print("loading Gemma3 ...")
    t0 = time.time()
    gemma = Gemma3()
    print(f"  loaded in {time.time() - t0:.1f}s")

    decoder = gemma.decoder
    snap = snapshot_linears(decoder)

    # --- Trustworthy inputs only (per tighten_finding) ---
    inputs: dict[str, torch.Tensor] = {}

    print("\ngenerating self-rollouts seed=2 ...")
    t0 = time.time()
    with _silent():
        self_ids = self_generated_rollouts(
            decoder, bos_id=2, num_seqs=2, length=32, temperature=1.0,
            seed=2, device=gemma.device,
        )
    print(f"  done in {time.time() - t0:.1f}s")
    try:
        preview = gemma.tokenizer.decode(self_ids[0].tolist())
        print(f"  preview: {preview[:120]!r}")
    except Exception:
        pass
    inputs["self_s2"] = self_ids

    real_tok_ids = gemma.tokenizer.encode(REAL_TEXT)[:32]
    while len(real_tok_ids) < 32:
        real_tok_ids.append(2)
    real_ids = torch.tensor([[2] + real_tok_ids[:31]],
                              dtype=torch.long, device=gemma.device)
    real_ids = real_ids.repeat(2, 1)
    inputs["real_text"] = real_ids
    print(f"\nreal-text sample: {gemma.tokenizer.decode(real_ids[0].tolist())[:120]!r}")

    # --- Score ---
    all_results: dict[str, list[dict]] = {}
    for label, ids in inputs.items():
        fp_states, fp_logits = _fp_capture(decoder, ids)
        all_results[label] = _score(decoder, ids, fp_states, fp_logits,
                                      snap, label)

    # --- Save ---
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = REPORT_DIR / "mxfp4_vs_rtn.json"
    out_json.write_text(json.dumps({
        "setup": {
            "model": "google/gemma-3-1b-it",
            "device": str(gemma.device),
            "configs": [c[0] for c in CONFIGS],
            "input_sets": list(inputs.keys()),
        },
        "results": all_results,
    }, indent=2))
    print(f"\nwrote {out_json}")

    # --- Summary ---
    print("\n=== KL table (rows=configs, cols=inputs) ===")
    header_inputs = list(inputs.keys())
    print(f"  {'config':12s}  " + "  ".join(f"{h:>11s}" for h in header_inputs))
    for cfg_name, _ in CONFIGS:
        row = []
        for h in header_inputs:
            kl = next(r["output_kl"] for r in all_results[h] if r["config"] == cfg_name)
            row.append(f"{kl:11.3e}")
        print(f"  {cfg_name:12s}  " + "  ".join(row))

    print("\n=== rankings (best -> worst) ===")
    for h in header_inputs:
        print(f"  {h:12s}: {_rank(all_results[h])}")

    # Direct head-to-head: mxfp4 vs rtn4_g32 (same block size)
    print("\n=== MXFP4 vs rtn4_g32 (same block-32 footprint) ===")
    for h in header_inputs:
        mx = next(r["output_kl"] for r in all_results[h] if r["config"] == "mxfp4")
        r32 = next(r["output_kl"] for r in all_results[h] if r["config"] == "rtn4_g32")
        winner = "MXFP4" if mx < r32 else "RTN-g32"
        ratio = max(mx, r32) / min(mx, r32)
        print(f"  {h:12s}: mxfp4 KL={mx:.3e}  rtn4_g32 KL={r32:.3e}  "
              f"-> {winner} wins ({ratio:.2f}x)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
