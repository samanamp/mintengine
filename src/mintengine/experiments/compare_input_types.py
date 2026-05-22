"""Test the predicted failure mode: does random-token MSE rank methods correctly?

Hypothesis: random tokens under-represent activation outliers, so quantization
methods whose value comes from outlier protection (per-group, AWQ-style scaling)
will look weak on random tokens but win on self-generated (on-manifold) inputs.

For each input type (random, self-generated from BOS), score the same set of
RTN configs and compare rankings.

Run:  uv run python -m mintengine.experiments.compare_input_types
"""

from __future__ import annotations

import contextlib
import io
import json
import time
from pathlib import Path

from mintengine.eval import (
    capture_layer_states,
    layer_mse,
    output_kl,
    random_tokens,
    self_generated_rollouts,
)
from mintengine.models.gemma3 import Gemma3
from mintengine.quant import apply_quant, restore_fp, snapshot_linears
from mintengine.quant.methods import RTN


REPORT_DIR = Path(__file__).resolve().parents[3] / "reports" / "quant"

CONFIGS = [
    ("rtn4_row", RTN(bits=4)),
    ("rtn4_g128", RTN(bits=4, group_size=128)),
    ("rtn4_g64", RTN(bits=4, group_size=64)),
    ("rtn4_g32", RTN(bits=4, group_size=32)),
]


@contextlib.contextmanager
def _silent():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def _score(decoder, ids, fp_states, fp_logits, label):
    print(f"  scoring on {label} (shape={tuple(ids.shape)}) ...")
    snap = snapshot_linears(decoder)
    results = []
    for name, method in CONFIGS:
        apply_quant(decoder, method)
        try:
            with _silent():
                q_states, q_logits = capture_layer_states(decoder, ids)
        finally:
            restore_fp(decoder, snap)
        kl = output_kl(fp_logits, q_logits)
        mses = layer_mse(fp_states, q_states)
        results.append({
            "config": name,
            "output_kl": kl,
            "max_mse": max(mses),
            "worst_layer": mses.index(max(mses)),
            "layer_mse": mses,
        })
        print(f"    {name:12s} KL={kl:.4e}  max_mse={max(mses):.2e}@L{mses.index(max(mses))}")
    return results


def main() -> int:
    print("loading Gemma3 ...")
    t0 = time.time()
    gemma = Gemma3()
    print(f"  loaded in {time.time() - t0:.1f}s")

    decoder = gemma.decoder
    vocab_size = gemma.gemma3_weights.embed_tokens.shape[0]

    # --- Random tokens
    random_ids = random_tokens(vocab_size, batch=2, length=32, seed=0,
                                device=gemma.device)

    # --- Self-generated from BOS (Gemma BOS = 2 per existing code)
    print("\ngenerating self-rollouts (BOS only, T=1.0) ...")
    t0 = time.time()
    with _silent():
        self_ids = self_generated_rollouts(
            decoder, bos_id=2, num_seqs=2, length=32, temperature=1.0,
            seed=0, device=gemma.device,
        )
    print(f"  generated {tuple(self_ids.shape)} in {time.time() - t0:.1f}s")
    # Sneak peek at one rollout
    try:
        toks = self_ids[0].tolist()
        print(f"  rollout[0] first 10 ids: {toks[:10]}")
        decoded = gemma.tokenizer.decode(toks)
        print(f"  rollout[0] decoded: {decoded[:200]!r}")
    except Exception as e:
        print(f"  (decode skipped: {e})")

    # FP captures
    print("\nFP captures ...")
    with _silent():
        fp_states_r, fp_logits_r = capture_layer_states(decoder, random_ids)
        fp_states_s, fp_logits_s = capture_layer_states(decoder, self_ids)

    print("\n=== random tokens ===")
    res_random = _score(decoder, random_ids, fp_states_r, fp_logits_r, "random")

    print("\n=== self-generated ===")
    res_self = _score(decoder, self_ids, fp_states_s, fp_logits_s, "self-generated")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = REPORT_DIR / "input_type_results.json"
    out_json.write_text(json.dumps({
        "setup": {
            "model": "google/gemma-3-1b-it",
            "device": str(gemma.device),
            "configs": [c[0] for c in CONFIGS],
            "random": {"batch": 2, "length": 32, "seed": 0},
            "self_generated": {"num_seqs": 2, "length": 32, "temperature": 1.0,
                                "seed": 0, "bos_id": 2},
        },
        "random": res_random,
        "self_generated": res_self,
    }, indent=2))
    print(f"\nwrote {out_json}")

    print("\n=== KL comparison: random vs self-generated ===")
    print(f"  {'config':12s}  {'random_KL':>10s}  {'self_KL':>10s}  {'ratio':>8s}")
    for r, s in zip(res_random, res_self):
        ratio = s["output_kl"] / r["output_kl"] if r["output_kl"] > 0 else float("inf")
        print(f"  {r['config']:12s}  {r['output_kl']:10.3e}  {s['output_kl']:10.3e}"
              f"  {ratio:8.2f}x")

    # Rank stability: if random ranks methods the same as self, ranking is reliable
    rank_random = sorted(range(len(CONFIGS)), key=lambda i: res_random[i]["output_kl"])
    rank_self = sorted(range(len(CONFIGS)), key=lambda i: res_self[i]["output_kl"])
    print("\nrank by KL (best -> worst):")
    print(f"  random:        {[CONFIGS[i][0] for i in rank_random]}")
    print(f"  self-generated:{[CONFIGS[i][0] for i in rank_self]}")
    print(f"  identical ranking: {rank_random == rank_self}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
