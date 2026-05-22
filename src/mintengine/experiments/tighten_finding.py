"""Make the random-vs-self-vs-real ranking story robust.

For each input type (random | self-generated x N seeds | real text), score
the same RTN configs and compare. The "real text" sample is the sanity floor:
if self-generated rollouts give similar rankings/magnitudes to real text, the
data-free signal is trustworthy.

Wider config grid than compare_input_types: more group sizes, plus row baseline,
so the monotonic trend (smaller group -> lower KL) shows clearly on the
trustworthy inputs and is disrupted on the untrustworthy ones.

Run:  uv run python -m mintengine.experiments.tighten_finding
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
    ("rtn4_g16", RTN(bits=4, group_size=16)),
    ("rtn4_g8", RTN(bits=4, group_size=8)),
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


def _score_all(decoder, ids, fp_states, fp_logits, snap, label):
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
        })
        print(f"  {name:11s} KL={kl:.4e}  max_mse={max(mses):.2e}@L{mses.index(max(mses))}")
    return out


def _fp_capture(decoder, ids):
    with _silent():
        states, logits = capture_layer_states(decoder, ids)
    return states, logits


def _rank(results):
    """Return list of config names sorted best (lowest KL) -> worst."""
    return [r["config"] for r in sorted(results, key=lambda r: r["output_kl"])]


def main() -> int:
    print("loading Gemma3 ...")
    t0 = time.time()
    gemma = Gemma3()
    print(f"  loaded in {time.time() - t0:.1f}s")

    decoder = gemma.decoder
    vocab_size = gemma.gemma3_weights.embed_tokens.shape[0]
    snap = snapshot_linears(decoder)

    # --- Build all input sets ---
    inputs: dict[str, torch.Tensor] = {}

    # Random
    inputs["random_s0"] = random_tokens(vocab_size, batch=2, length=32, seed=0,
                                         device=gemma.device)
    inputs["random_s1"] = random_tokens(vocab_size, batch=2, length=32, seed=1,
                                         device=gemma.device)

    # Self-generated, 3 seeds
    for seed in (0, 1, 2):
        print(f"\ngenerating self-rollouts seed={seed} ...")
        t0 = time.time()
        with _silent():
            ids = self_generated_rollouts(
                decoder, bos_id=2, num_seqs=2, length=32, temperature=1.0,
                seed=seed, device=gemma.device,
            )
        print(f"  done in {time.time() - t0:.1f}s")
        try:
            preview = gemma.tokenizer.decode(ids[0].tolist())
            print(f"  preview: {preview[:100]!r}")
        except Exception:
            pass
        inputs[f"self_s{seed}"] = ids

    # Real text, tokenized to 32 tokens
    real_tok_ids = gemma.tokenizer.encode(REAL_TEXT)[:32]
    while len(real_tok_ids) < 32:  # pad with BOS-clone in case tokenizer is short
        real_tok_ids.append(2)
    real_ids = torch.tensor([[2] + real_tok_ids[:31]],
                              dtype=torch.long, device=gemma.device)
    # batch=2 so the comparison is apples-to-apples with the others
    real_ids = real_ids.repeat(2, 1)
    inputs["real_text"] = real_ids
    print(f"\nreal-text sample: {gemma.tokenizer.decode(real_ids[0].tolist())[:120]!r}")

    # --- Score on each input set ---
    all_results: dict[str, list[dict]] = {}
    for label, ids in inputs.items():
        fp_states, fp_logits = _fp_capture(decoder, ids)
        all_results[label] = _score_all(decoder, ids, fp_states, fp_logits,
                                          snap, label)

    # --- Summary ---
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = REPORT_DIR / "tighten_results.json"
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

    # Disagreement between random and "trustworthy" (self + real)
    real_rank = _rank(all_results["real_text"])
    print(f"\n=== rank disagreement vs real_text (Spearman-ish) ===")
    for h in header_inputs:
        r = _rank(all_results[h])
        # count swapped pairs
        swaps = sum(
            1
            for i in range(len(r))
            for j in range(i + 1, len(r))
            if (real_rank.index(r[i]) > real_rank.index(r[j]))
        )
        print(f"  {h:12s}: {swaps} inverted pairs (out of {len(r)*(len(r)-1)//2})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
