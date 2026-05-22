# Input distribution flips method rankings in data-free quantization eval

A naive autonomous-research loop using random-token inputs to rank quantization
methods will **pick the wrong winner**. The same set of methods, scored against
self-generated rollouts from the same model, gives a different (and correct)
ranking with KL scores 2–9× smaller.

## Setup

- Model: `google/gemma-3-1b-it`, 26 layers
- Methods: per-output-row vs per-group RTN at 4 bits; groups ∈ {row, 128, 64, 32}
- Inputs:
  - **random**: `randint(0, vocab_size, (2, 32))`
  - **self-generated**: 2 sequences of length 32, sampled with `T=1.0`
    starting from `[BOS]` only
- Metric: end-to-end `KL(P_fp ‖ P_quant)`
- Run: `uv run python -m mintengine.experiments.compare_input_types`

## Result

| config       | random KL  | self-gen KL | self/random | rank (random) | rank (self) |
| ------------ | ---------- | ----------- | ----------- | ------------- | ----------- |
| `rtn4_row`   | 5.339      | 2.531       | 0.47×       | 4 (worst)     | 4 (worst)   |
| `rtn4_g128`  | 4.307      | 1.254       | 0.29×       | 2             | 3           |
| `rtn4_g64`   | **4.101**  | 1.035       | 0.25×       | **1**         | 2           |
| `rtn4_g32`   | 4.346      | **0.502**   | **0.12×**   | 3             | **1**       |

**Random tokens get the winner wrong.** They rank `g64` first and `g32` third.
Self-generated rollouts produce the expected monotonic order `g32 > g64 > g128 > row`.

## Why this happens — mechanism

Per-group RTN improves over per-row precisely by giving large activation outliers
their own scale factor (the rest of the row no longer has to share dynamic range
with one outsized element). The "win" therefore *only shows up if outliers exist
in the inputs being measured.*

Random tokens produce activations whose distribution is shaped by the model's
embedding + early-layer dynamics on uniform-random ids. They do not exercise
long-range attention patterns or natural-text token co-occurrence statistics,
which are precisely what create the heavy-tailed activation outliers documented
in LLM.int8, SmoothQuant, AWQ, and QuaRot.

Self-generated rollouts (model talks to itself from `[BOS]`, even with totally
synthetic-looking content like the Cyrillic-code-comment we got at T=1.0) put
the activations back on the manifold the model actually inhabits. Outliers
return. Per-group's scale-factor budget pays off.

## Implication for autonomous quantization research

**The inner-loop scoring signal is a load-bearing decision.** If a multi-agent
exploration loop uses random-token layer-MSE to triage candidate methods, it
will systematically reject methods that target outliers — i.e. exactly the
research direction that has produced the strongest results in the field over
the last two years. That's not a small bug; it inverts the search.

Recommended scoring stack:

1. **Inner loop** (per-candidate ranking, milliseconds):
   self-generated rollouts from `[BOS]` × small `num_seqs` × short `length` →
   layer-MSE for diagnostic per-layer breakdown.
2. **Outer loop** (final acceptance, seconds):
   same rollouts at longer `length` → output KL as arbiter.
3. **Sanity floor:** verify on a held-out real-text sample before any claim
   of "this method works."

A single set of rollouts can serve all three — generate once, score many.

## A side observation about per-layer MSE

The `rtn4_baseline.md` writeup pointed to a layer-7 MSE cliff and conjectured
"weight-distribution outliers in layer 7." The Phase-A ablations from the
sweep run (`sweep_results.json`) instead show:

- `rtn4_skip_L7`: KL 5.03 (vs full 5.34) — protecting layer 7 helps almost nothing
- `rtn4_only_L7`: KL 0.76 (vs full 5.34) — breaking only layer 7 only mildly degrades
- `rtn4_skip_L7+L8+L9`: KL 4.61 — even skipping the three worst-MSE layers barely helps

So the **high per-layer MSE at layer 7 was a manifestation, not a cause**:
accumulated upstream noise crossed a threshold there. The actual damage is
distributed across most layers, with no single dominant culprit. Per-layer
MSE is a useful diagnostic but not a reliable causal localizer — another
reason the outer-loop KL is the arbiter.

## Generated artifacts

- `reports/quant/sweep_results.json` — full bits × group × layer-skip sweep
- `reports/quant/input_type_results.json` — random vs self-generated comparison
- `src/mintengine/experiments/sweep_rtn.py`
- `src/mintengine/experiments/compare_input_types.py`
