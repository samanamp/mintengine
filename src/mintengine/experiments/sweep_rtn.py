"""Single-load sweep over RTN configurations on Gemma3-1B.

Phase A — diagnose the layer-7 cliff observed in rtn4_baseline:
  * skip_layer7  : quantize all except layer 7 (does protecting it alone help?)
  * only_layer7  : quantize only layer 7 (does breaking it alone reproduce?)
  * skip_top3    : protect the 3 worst-MSE layers from the baseline (7, 8, 9)

Phase B — well-known wins:
  * bits sweep   : 8, 6, 4, 3 at per-row
  * group sweep  : 4-bit with group_size in {None, 128, 64, 32}

All on the same model load to make comparisons clean. Each config:
  1. snapshot_linears(decoder) cached once on init
  2. apply_quant(decoder, method, where=filter)
  3. capture_layer_states
  4. restore_fp(decoder, snapshot)

Writes a structured JSON to reports/quant/sweep_results.json plus a markdown
summary alongside it.

Run:  uv run python -m mintengine.experiments.sweep_rtn
"""

from __future__ import annotations

import contextlib
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from mintengine.eval import capture_layer_states, layer_mse, output_kl, random_tokens
from mintengine.models.gemma3 import Gemma3
from mintengine.quant import apply_quant, restore_fp, snapshot_linears
from mintengine.quant.linear import WhereFn
from mintengine.quant.methods import RTN


REPORT_DIR = Path(__file__).resolve().parents[3] / "reports" / "quant"


@dataclass
class Config:
    label: str
    method: RTN
    where: WhereFn | None = None
    note: str = ""


def _build_configs() -> list[Config]:
    def in_layer(i: int) -> WhereFn:
        prefix = f"layers.{i}."
        return lambda path, _: prefix in path

    def not_in_layers(ids: set[int]) -> WhereFn:
        prefixes = tuple(f"layers.{i}." for i in ids)
        return lambda path, _: not any(p in path for p in prefixes)

    cfgs: list[Config] = []
    # --- Phase A: diagnose layer-7 cliff
    cfgs.append(Config("rtn4_skip_L7", RTN(bits=4), not_in_layers({7}),
                       "quantize all but layer 7"))
    cfgs.append(Config("rtn4_only_L7", RTN(bits=4), in_layer(7),
                       "quantize only layer 7"))
    cfgs.append(Config("rtn4_skip_L7_L8_L9", RTN(bits=4), not_in_layers({7, 8, 9}),
                       "skip the 3 worst layers from baseline"))
    # --- Phase B: bits sweep (per-row)
    for b in (8, 6, 4, 3):
        cfgs.append(Config(f"rtn{b}_row", RTN(bits=b), None, "full model, per-row"))
    # --- Phase B: group sweep at 4 bits
    for g in (128, 64, 32):
        cfgs.append(Config(f"rtn4_g{g}", RTN(bits=4, group_size=g),
                           None, f"full model, group_size={g}"))
    return cfgs


@contextlib.contextmanager
def _silent_stdout():
    """Swallow the Gemma3Layer debug prints during a forward pass."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


def _evaluate(decoder, input_ids, fp_states, fp_logits):
    with _silent_stdout():
        q_states, q_logits = capture_layer_states(decoder, input_ids)
    return {
        "output_kl": output_kl(fp_logits, q_logits),
        "layer_mse": layer_mse(fp_states, q_states),
    }


def main() -> int:
    print("loading Gemma3 ...")
    t0 = time.time()
    gemma = Gemma3()
    print(f"  loaded in {time.time() - t0:.1f}s")

    decoder = gemma.decoder
    vocab_size = gemma.gemma3_weights.embed_tokens.shape[0]
    ids = random_tokens(vocab_size, batch=2, length=32, seed=0, device=gemma.device)

    snap = snapshot_linears(decoder)
    print(f"  snapshot: {len(snap)} Linear modules")

    print("\nFP baseline capture ...")
    t0 = time.time()
    with _silent_stdout():
        fp_states, fp_logits = capture_layer_states(decoder, ids)
    print(f"  done in {time.time() - t0:.1f}s")

    configs = _build_configs()
    results = []
    for i, cfg in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] {cfg.label:25s}  {cfg.note}")
        t0 = time.time()
        apply_quant(decoder, cfg.method, where=cfg.where)
        try:
            out = _evaluate(decoder, ids, fp_states, fp_logits)
        finally:
            restore_fp(decoder, snap)
        elapsed = time.time() - t0
        max_mse = max(out["layer_mse"])
        worst_layer = out["layer_mse"].index(max_mse)
        print(
            f"   KL={out['output_kl']:.3e}   max_mse={max_mse:.2e}@L{worst_layer}"
            f"   ({elapsed:.1f}s)"
        )
        results.append({
            "label": cfg.label,
            "method_name": cfg.method.name,
            "note": cfg.note,
            "output_kl": out["output_kl"],
            "layer_mse": out["layer_mse"],
            "max_mse": max_mse,
            "worst_layer": worst_layer,
            "elapsed_s": elapsed,
        })

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = REPORT_DIR / "sweep_results.json"
    out_json.write_text(json.dumps({
        "setup": {
            "model": "google/gemma-3-1b-it",
            "batch": 2,
            "length": 32,
            "seed": 0,
            "device": str(gemma.device),
            "num_layers": len(fp_states),
        },
        "results": results,
    }, indent=2))
    print(f"\nwrote {out_json}")

    print("\n=== summary (sorted by output KL) ===")
    for r in sorted(results, key=lambda r: r["output_kl"]):
        print(f"  {r['label']:25s} KL={r['output_kl']:.3e}   "
              f"max_mse={r['max_mse']:.2e}@L{r['worst_layer']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
