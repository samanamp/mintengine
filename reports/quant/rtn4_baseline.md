# RTN 4-bit baseline on Gemma3-1B-IT

**Setup:** per-output-row symmetric round-to-nearest, fake-quantized (dequant
weight, fp matmul). Inputs: 2 random-token sequences, length 32, seed 0.
Signal = layer-boundary state `x + residual` after each `Gemma3Layer`.

**Run:** `uv run python -m mintengine.experiments.compare_rtn --bits 4 --batch 2 --length 32`

## Headline

| metric | value |
| --- | --- |
| output KL(FP ‖ RTN4) | **5.34** |
| max per-layer MSE | 3.95e+04 (layer 25) |
| layer where MSE jumps >10× in one step | **layer 7** (154 → 2249) |

## Per-layer MSE

```
layer  0: 3.22e+00
layer  1: 4.91e+00
layer  2: 1.09e+01
layer  3: 2.64e+01
layer  4: 3.69e+01
layer  5: 8.89e+01     <- global-attention layer (every 6th)
layer  6: 1.54e+02
layer  7: 2.25e+03     <- cliff: ~15x jump in a single layer
layer  8: 4.03e+03
layer  9: 7.33e+03
layer 10: 1.07e+04
layer 11: 1.51e+04     <- global-attention layer
layer 12: 1.99e+04
layer 13: 2.32e+04
layer 14: 2.60e+04
layer 15: 2.75e+04
layer 16: 3.13e+04
layer 17: 3.52e+04     <- global-attention layer
layer 18: 3.65e+04
layer 19: 3.72e+04
layer 20: 3.29e+04
layer 21: 3.31e+04
layer 22: 2.92e+04
layer 23: 2.57e+04     <- global-attention layer
layer 24: 2.69e+04
layer 25: 3.95e+04
```

## Observations

1. **Per-row 4-bit RTN destroys the output distribution** (KL≈5.3). Expected —
   per-row symmetric RTN at 4-bit is a known-weak baseline; this is the
   "floor" we're trying to beat.
2. **The layer-7 cliff is the headline diagnostic.** The harness pays for
   itself the first time it runs: instead of "the model broke," we see
   *exactly* where it broke. Plausible suspects:
   - Activation outliers concentrated around mid-stack (LLM.int8 /
     SmoothQuant / AWQ all target this regime).
   - Doesn't align with the sliding/global attention boundary (layers
     5, 11, 17, 23) — so probably weight-distribution, not attention-pattern.
3. **Late-layer plateau (~3e+04)** suggests saturation: once the hidden
   state has diverged this far, individual layers can only redistribute,
   not amplify further. Not a hopeful sign for downstream quality but
   useful to know the geometry.

## Next experiments this enables

- Same plot at bits ∈ {8, 6, 4, 3, 2} — find the cliff in bit-space.
- Per-group RTN (group_size=128) — should soften the layer-7 cliff if
  outliers are the cause.
- Per-layer skip (quantize all but layer 7) — sanity check that the
  cliff layer alone is doing most of the damage.
- Activation-aware variant (AWQ-style scaling) — direct test of the
  outlier hypothesis.
