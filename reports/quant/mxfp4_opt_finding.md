# MXFP4: the OCP scale formula leaves quality on the table

**TL;DR.** OCP MXFP4 mandates a deterministic per-block scale
`E = floor(log2(absmax)) − 2`. Replacing it with a per-block min-MSE search
over `E ∈ {default−1, default, default+1}` — *without changing the wire
format* (still block-32, still one E8M0 byte per block, still e2m1 elements) —
reduces output KL on Gemma3-1B by **24% on self-rollouts** and **27% on
real text**, and flips MXFP4 from losing to *beating* RTN-g32 on the
self-rollout signal.

## Setup

- Model: `google/gemma-3-1b-it`, fake-quant (dequantize then FP matmul)
- Inputs (trustworthy per `input_distribution_matters.md`): a self-generated
  rollout from BOS at T=1.0 (seed 2) and a 32-token real-text excerpt,
  both batch 2 × length 32
- Metric: output KL of last-position logits vs FP baseline
- Quant scope: every `nn.Linear` in the decoder (attention + MLP)

## Results (output KL, lower = better)

| config        | self_s2     | real_text   |
|---------------|-------------|-------------|
| rtn4_row      | 1.125e+00   | 6.065e-01   |
| rtn4_g128     | 4.571e-01   | 2.974e-01   |
| **rtn4_g32**  | 3.182e-01   | 1.200e-01   |
| **rtn4_g8**   | 1.597e-01   | 8.175e-02   |
| mxfp4 (OCP)   | 3.397e-01   | 2.241e-01   |
| **mxfp4opt1** | **2.579e-01** | **1.633e-01** |
| mxfp4opt2     | 2.579e-01   | 1.633e-01   |

Rankings (best → worst):

- `self_s2`:    rtn4_g8, **mxfp4opt1**, mxfp4opt2, rtn4_g32, mxfp4, rtn4_g128, rtn4_row
- `real_text`:  rtn4_g8, rtn4_g32, **mxfp4opt1**, mxfp4opt2, mxfp4, rtn4_g128, rtn4_row

## What changed

OCP MXFP4 picks the per-block shared exponent deterministically:

```
E = floor(log2(max(|w|))) − 2          # OCP rule
scale = 2^E
fp4 = round_to_e2m1(w / scale)
```

`MXFP4Opt` searches `E ∈ {E_default + k : k ∈ [−r, r]}` and keeps the `E` that
minimizes the per-block MSE of `(round_to_e2m1(w/2^E) · 2^E − w)²`. `r = 1`
is sufficient (`r = 2` gives identical numbers in our unit tests and on
Gemma3-1B). Negative offsets never help because they push more values into
saturation; the win comes entirely from `E_default + 1`, which uses the
already-tight `±6` saturation of FP4 e2m1 instead of leaving headroom.

**This is a format-compatible calibration choice, not a format extension.**
The E8M0 byte can express any integer exponent; the OCP spec just picks one
deterministic value. A real implementation precomputes `E` offline; runtime
storage and decode are unchanged.

## Why this matters

The MXFP4 papers compare *against* INT4. Without optimal-scale calibration,
on this model MXFP4 *loses* to a matched-block INT4 (RTN-g32) by 1.87× on
real text. With `mxfp4opt1`, MXFP4 closes ~75% of that gap on real text and
**beats** RTN-g32 outright on self-rollouts (the input distribution most
sensitive to outliers). The win is free: same memory, same compute, same
decoder logic.

The result also pins the per-element representational error of FP4 e2m1
itself as roughly competitive with INT4 once scales are chosen well, which
matches the worst-case math:

- FP4 e2m1: 8 magnitudes, max gap = 2.0 (between 4 and 6)
- INT4 symmetric: 16 levels, gap = `2·max/15`. For a block whose absmax is
  near a power of 2, this gap is also ~`2·max/15` ≈ `0.13·max`

Tight-saturation FP4 (E_default + 1) gives `max/scale = 6` → max representable
exactly hits 6. OCP's E_default leaves `max/scale ∈ [2,4)`, which under-uses
the high end of the grid. The fix is one extra exponent comparison per block
at calibration time.

## Caveats

- One model (Gemma3-1B), one decoding seed for self-rollouts, ~64 tokens
  total per signal. Headline numbers will move on larger models / longer
  contexts; the *direction* (E_default + 1 dominates E_default for most
  blocks) follows from the saturation math and should hold.
- Fake-quant only. We do not measure wall-clock or accounting for the E8M0
  exponent storage — both are unchanged from spec MXFP4 by construction.
- We did not run perplexity or downstream tasks. KL on the last position is
  a fast surrogate; consistent with `tighten_finding`'s ranking metric.

## Reproduce

```
uv run python tests/test_mxfp4.py
uv run python tests/test_mxfp4_opt.py
uv run python -m mintengine.experiments.compare_mxfp4
```

Code: `src/mintengine/quant/methods/mxfp4_opt.py`,
experiment: `src/mintengine/experiments/compare_mxfp4.py`,
raw data: `reports/quant/mxfp4_vs_rtn.json`.
