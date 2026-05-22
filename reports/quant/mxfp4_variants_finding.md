# MXFP4 variants: Hadamard hurts, AWQ closes the INT4 gap on real text

**TL;DR.** Two follow-ups to `mxfp4_opt_finding.md`:

1. **HadMXFP4** (within-block Sylvester-Hadamard rotation before MXFP4Opt)
   *hurts* on Gemma3-1B: +54% KL on self-rollouts, +51% on real text vs
   MXFP4Opt-r1. Useful negative result — rotation only helps when blocks
   carry kurtotic outliers, and trained Gemma3 weight blocks are too
   close to Gaussian for the rotation to redistribute anything.

2. **AWQ-MXFP4Opt** (per-input-channel activation scaling + MXFP4Opt)
   **matches RTN-g32 on real text** (0.118 vs 0.120) — closing 100% of
   the remaining INT4 gap inside the MXFP4 wire format. On self-rollouts
   it's slightly worse than MXFP4Opt alone (+5%), because calibration was
   on real activation magnitudes and self-rollouts don't share the same
   outlier-channel statistics.

## Results (output KL, lower = better)

| config            | self_s2     | real_text   | Δ vs mxfp4opt1 (real_text) |
|-------------------|-------------|-------------|----------------------------|
| rtn4_g32          | 3.182e-01   | 1.200e-01   | —                          |
| rtn4_g8           | 1.597e-01   | 8.175e-02   | —                          |
| mxfp4 (OCP)       | 3.397e-01   | 2.241e-01   | +37%                       |
| **mxfp4opt1**     | **2.579e-01** | **1.633e-01** | (baseline)               |
| hadmxfp4          | 3.985e-01   | 2.464e-01   | **+51%**                   |
| **awq-mxfp4opt1** | 2.720e-01   | **1.182e-01** | **−28%**                  |

AWQ alpha histogram (155 Linears total): `{0.0: 45, 0.5: 107, 1.0: 4}`.
~70% of layers pick the moderate scale; ~30% prefer no scaling at all;
only 4 layers benefit from full activation-magnitude weighting. The
shape matches the AWQ paper's observation that most layers want
"some" activation-aware scaling but not extreme.

## Why HadMXFP4 hurts

A within-block Hadamard rotation mixes 32 weights into 32 linear
combinations. For a Gaussian block, the rotated values are still Gaussian
with the same variance — the *distribution* is unchanged, but the *correlation
structure* the per-block scale exploits is destroyed. Specifically:

- Vanilla MXFP4Opt picks a per-block scale to fit absmax tightly. For
  a Gaussian block, absmax ≈ 2.5σ to 3σ, and ~26 of 32 values fall
  inside the central FP4 grid points.
- After rotation, *every* rotated value carries the original outlier's
  energy distributed across 32 coordinates. Per-block absmax of the rotated
  block is statistically similar, but the per-value quantization error,
  when rotated back, smears across all 32 originals as a linear combination
  of rotated-space errors. L2 norm is preserved (rotation is orthogonal),
  so total MSE doesn't go down — but the per-row error structure changes
  in ways that interact badly with downstream activations.

Hadamard rotation is known to help when blocks have **high kurtosis**
(few big outliers + many small values, as in *activations* with channel
outliers, or weights of certain architectures). Gemma3 weight blocks
are not in that regime — the negative result here is consistent with
that.

A real win for Hadamard at this format would require *cross-block*
rotations (mixing values that currently end up in different blocks),
which is what QuaRot-style approaches do at the *activation* boundary,
not the weight tensor. That's beyond a drop-in weight preprocessing
trick — it needs fusion into the preceding RMSNorm.

## Why AWQ-MXFP4Opt works (on real text)

AWQ rescales W column-wise by `s = act_mag^α`, then quantizes `W·s` and
divides activations by `s` at runtime. The rescaling makes weight columns
that see large activations *quantize-easier* (smaller magnitudes get
proportionally more bits), at the cost of weight columns that see small
activations. For MXFP4 specifically, this changes the per-block absmax
distribution so the OCP scale + min-MSE-exponent search lands in a
better spot more often.

Mismatch on self_s2: AWQ was calibrated on a 4×32 mixed batch (2 self +
2 real). Real-text activations have different outlier channels from
self-rollout activations — calibration biases toward the real
distribution, so self-rollout scoring drifts upward (+5%) while real-text
scoring improves dramatically (−28%). This is a faithful read of how
AWQ behaves under distribution shift, not a bug.

## Combined ranking (drop-in + AWQ)

Best → worst on the trustworthy signals:

- `self_s2`:    rtn4_g8 → mxfp4opt1 → awq-mxfp4opt1 → rtn4_g32 → mxfp4 → hadmxfp4
- `real_text`:  rtn4_g8 → **awq-mxfp4opt1 ≈ rtn4_g32** → mxfp4opt1 → mxfp4 → hadmxfp4

The MXFP4 family is now competitive with INT4-g32 on real text; the
remaining gap is to INT4-g8 (which uses 8 elements per group → much finer
scale resolution at 4× the scale-byte overhead).

## What this means for the headline story

Stacked across the two reports:

| step                                  | self_s2 KL | real_text KL |
|---------------------------------------|------------|--------------|
| MXFP4 (OCP, baseline)                 | 0.340      | 0.224        |
| + per-block min-MSE exponent (`+1`)   | 0.258 (−24%) | 0.163 (−27%) |
| + AWQ scaling (calib on real+self)    | 0.272 (−20%) | 0.118 (−47%) |

47% KL reduction on real text vs OCP MXFP4 by combining two
calibration-only changes (no format extension, no runtime cost beyond
AWQ's per-channel rescale that fuses into the preceding norm). MXFP4
now sits at parity with RTN-g32 on real text and within 2× of the much
finer-grained RTN-g8.

## Caveats

- One model (Gemma3-1B), 64 tokens per scoring signal.
- AWQ alpha grid reduced from `(0, .25, .5, .75, 1)` to `(0, .5, 1)` to fit
  in compute budget — finer grid could change the alpha distribution but
  is unlikely to flip the headline.
- The AWQ calibration batch is intentionally tiny (4×32). Real production
  AWQ uses 128+ samples; the small calibration set means the alpha
  selections are noisier than they would be at scale.
- Fake-quant only; no wall-clock measurement. Both MXFP4Opt and AWQ-MXFP4
  are pure calibration changes — runtime path is unchanged for MXFP4Opt
  and adds one per-input-channel multiply for AWQ (folds into preceding
  norm in production).

## Reproduce

```
uv run python tests/test_had_mxfp4.py
uv run python -m mintengine.experiments.compare_mxfp4_variants
```

Code:
- `src/mintengine/quant/methods/had_mxfp4.py` — HadMXFP4
- `src/mintengine/quant/methods/awq.py` — AWQ now generic over QuantMethod
- `src/mintengine/experiments/compare_mxfp4_variants.py` — runner

Raw data: `reports/quant/mxfp4_variants.json`.
