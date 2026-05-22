"""MXFP4 variants head-to-head: vanilla, opt, Hadamard, AWQ-wrapped.

Builds on compare_mxfp4 by adding two activation/representation-aware
variants on top of MXFP4Opt-r1:

  - HadMXFP4-r1: within-block Sylvester-Hadamard rotation before MXFP4Opt
    (calibration-only rotation, undone in the dequantized weight — drop-in
    weight matrix, no runtime change).

  - AWQ + MXFP4Opt: per-input-channel scaling chosen by AWQ alpha-grid
    search on calibration activations, then MXFP4Opt on the scaled weights.
    Runtime path: y = (x * inv_scale) @ W_q.T, so activation rescale must
    persist (AWQLinear holds inv_scale buffer).

Scored on the trustworthy signals from tighten_finding (self_s2 + real_text).

Run:  uv run python -m mintengine.experiments.compare_mxfp4_variants
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
from mintengine.quant.methods import HadMXFP4, MXFP4, MXFP4Opt, RTN
from mintengine.quant.methods.awq import AWQConfig, apply_awq


REPORT_DIR = Path(__file__).resolve().parents[3] / "reports" / "quant"

# Drop-in QuantMethod configs (used via apply_quant).
DROPIN_CONFIGS = [
    ("rtn4_g32",   RTN(bits=4, group_size=32)),
    ("rtn4_g8",    RTN(bits=4, group_size=8)),
    ("mxfp4",      MXFP4(block_size=32)),
    ("mxfp4opt1",  MXFP4Opt(block_size=32, search_radius=1)),
    ("hadmxfp4",   HadMXFP4(block_size=32, search_radius=1)),
]

# AWQ-style configs (applied with calibration, replace Linear with AWQLinear).
# Kept to just MXFP4Opt because:
#   - AWQ alpha search runs MXFP4Opt.pack 5x per linear → expensive on CPU/MPS
#   - HadMXFP4 already lost as a drop-in; combining with AWQ is unlikely to flip
AWQ_CONFIGS = [
    ("awq-mxfp4opt1",
     AWQConfig(method=MXFP4Opt(block_size=32, search_radius=1),
               alpha_grid=(0.0, 0.5, 1.0))),  # 3 alphas instead of 5 to cut time
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


def _score_dropin(decoder, ids, fp_states, fp_logits, snap, label):
    out = []
    for name, method in DROPIN_CONFIGS:
        t0 = time.time()
        apply_quant(decoder, method)
        try:
            with _silent():
                q_states, q_logits = capture_layer_states(decoder, ids)
        finally:
            restore_fp(decoder, snap)
        kl = output_kl(fp_logits, q_logits)
        mses = layer_mse(fp_states, q_states)
        dt = time.time() - t0
        out.append({
            "config": name,
            "output_kl": kl,
            "max_mse": max(mses),
            "worst_layer": mses.index(max(mses)),
        })
        print(f"  {name:14s} KL={kl:.4e}  max_mse={max(mses):.2e}@L{mses.index(max(mses))}  ({dt:.1f}s)")
    return out


def _run_awq_once(decoder, calib_ids, score_inputs, fp_per_input, snap):
    """Calibrate AWQ once per config, score on all input sets, then restore.

    score_inputs : {label: ids}
    fp_per_input : {label: (fp_states, fp_logits)}
    Returns {label: [result dict, ...]} accumulating one entry per AWQ config.
    """
    out_by_input: dict[str, list[dict]] = {lbl: [] for lbl in score_inputs}
    for name, cfg in AWQ_CONFIGS:
        t0 = time.time()
        print(f"  [AWQ] calibrating {name} on {tuple(calib_ids.shape)} ...", flush=True)
        with _silent():
            chosen = apply_awq(decoder, calib_ids, cfg)
        dt_cal = time.time() - t0
        alphas = list(chosen.values())
        alpha_hist = {a: alphas.count(a) for a in sorted(set(alphas))}
        print(f"  [AWQ] {name} calibrated in {dt_cal:.1f}s, alphas={alpha_hist}", flush=True)

        try:
            for lbl, ids in score_inputs.items():
                ts = time.time()
                with _silent():
                    q_states, q_logits = capture_layer_states(decoder, ids)
                fp_states, fp_logits = fp_per_input[lbl]
                kl = output_kl(fp_logits, q_logits)
                mses = layer_mse(fp_states, q_states)
                dt_score = time.time() - ts
                out_by_input[lbl].append({
                    "config": name,
                    "output_kl": kl,
                    "max_mse": max(mses),
                    "worst_layer": mses.index(max(mses)),
                    "alpha_hist": alpha_hist,
                })
                print(f"  [AWQ] {name} on {lbl:12s} KL={kl:.4e}  "
                      f"max_mse={max(mses):.2e}@L{mses.index(max(mses))}  ({dt_score:.1f}s)",
                      flush=True)
        finally:
            restore_fp(decoder, snap)
    return out_by_input


def main() -> int:
    print("loading Gemma3 ...")
    t0 = time.time()
    gemma = Gemma3()
    print(f"  loaded in {time.time() - t0:.1f}s")

    decoder = gemma.decoder
    snap = snapshot_linears(decoder)

    # --- Inputs ---
    print("\ngenerating self-rollouts seed=2 ...")
    t0 = time.time()
    with _silent():
        self_ids = self_generated_rollouts(
            decoder, bos_id=2, num_seqs=2, length=32, temperature=1.0,
            seed=2, device=gemma.device,
        )
    print(f"  done in {time.time() - t0:.1f}s")

    real_tok_ids = gemma.tokenizer.encode(REAL_TEXT)[:32]
    while len(real_tok_ids) < 32:
        real_tok_ids.append(2)
    real_ids = torch.tensor([[2] + real_tok_ids[:31]],
                              dtype=torch.long, device=gemma.device).repeat(2, 1)

    # AWQ calibration uses both inputs concatenated → richer activation stats.
    calib_ids = torch.cat([self_ids, real_ids], dim=0)
    print(f"\ncalib batch for AWQ: {tuple(calib_ids.shape)}")

    inputs = {"self_s2": self_ids, "real_text": real_ids}

    # FP captures up-front, keyed by input label.
    fp_per_input: dict[str, tuple] = {}
    for label, ids in inputs.items():
        print(f"\nFP capture for {label}  shape={tuple(ids.shape)}", flush=True)
        fp_per_input[label] = _fp_capture(decoder, ids)

    # Drop-in methods: re-applied per input.
    all_results: dict[str, list[dict]] = {label: [] for label in inputs}
    for label, ids in inputs.items():
        print(f"\n--- drop-in on {label} ---", flush=True)
        fp_states, fp_logits = fp_per_input[label]
        all_results[label].extend(
            _score_dropin(decoder, ids, fp_states, fp_logits, snap, label)
        )

    # AWQ: calibrate once per config, score on all inputs.
    print(f"\n--- AWQ (calibrate once, score on all inputs) ---", flush=True)
    awq_results = _run_awq_once(decoder, calib_ids, inputs, fp_per_input, snap)
    for label in inputs:
        all_results[label].extend(awq_results[label])

    # --- Save ---
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = REPORT_DIR / "mxfp4_variants.json"
    out_json.write_text(json.dumps({
        "setup": {
            "model": "google/gemma-3-1b-it",
            "device": str(gemma.device),
            "dropin_configs": [c[0] for c in DROPIN_CONFIGS],
            "awq_configs": [c[0] for c in AWQ_CONFIGS],
            "input_sets": list(inputs.keys()),
            "calib_shape": list(calib_ids.shape),
        },
        "results": all_results,
    }, indent=2))
    print(f"\nwrote {out_json}")

    # --- Summary table ---
    print("\n=== KL table ===")
    all_cfg_names = [c[0] for c in DROPIN_CONFIGS] + [c[0] for c in AWQ_CONFIGS]
    header_inputs = list(inputs.keys())
    print(f"  {'config':16s}  " + "  ".join(f"{h:>11s}" for h in header_inputs))
    for cfg_name in all_cfg_names:
        row = []
        for h in header_inputs:
            kl = next(r["output_kl"] for r in all_results[h] if r["config"] == cfg_name)
            row.append(f"{kl:11.3e}")
        print(f"  {cfg_name:16s}  " + "  ".join(row))

    # --- Headline deltas vs MXFP4-opt-r1 (the established baseline) ---
    print("\n=== % change vs mxfp4opt1 ===")
    compare_to_baseline = [c[0] for c in DROPIN_CONFIGS if c[0] != "mxfp4opt1"] + \
                          [c[0] for c in AWQ_CONFIGS]
    for h in header_inputs:
        base = next(r["output_kl"] for r in all_results[h] if r["config"] == "mxfp4opt1")
        print(f"  {h}:")
        for cfg_name in compare_to_baseline:
            v = next((r["output_kl"] for r in all_results[h]
                       if r["config"] == cfg_name), None)
            if v is None:
                continue
            delta = (v - base) / base * 100
            sign = "+" if delta >= 0 else ""
            print(f"    {cfg_name:16s}  {v:.3e}  ({sign}{delta:.1f}%)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
