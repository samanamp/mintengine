"""Compare FP vs RTN on real Gemma3 with random-token rollouts.

Loads the model once, captures FP layer states + logits, applies RTN in-place,
captures quantized layer states + logits, then prints per-layer MSE and the
end-to-end output KL.

Usage:
    uv run python -m mintengine.experiments.compare_rtn --bits 4 --length 64

First run will download ~2GB of Gemma-3-1B weights from HF.

Known noise: Gemma3Layer.forward has print() debug statements that fire during
forward passes (gemma3.py:276+); the eval output is still readable but loud.
"""

from __future__ import annotations

import argparse

from mintengine.eval import (
    capture_layer_states,
    layer_mse,
    output_kl,
    random_tokens,
)
from mintengine.models.gemma3 import Gemma3
from mintengine.quant import apply_quant
from mintengine.quant.methods import RTN


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--length", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = _parse()
    print(f"loading Gemma3 ...")
    gemma = Gemma3()
    decoder = gemma.decoder
    vocab_size = gemma.gemma3_weights.embed_tokens.shape[0]

    print(f"\nrandom tokens: batch={args.batch} length={args.length}")
    ids = random_tokens(
        vocab_size=vocab_size,
        batch=args.batch,
        length=args.length,
        seed=args.seed,
        device=gemma.device,
    )

    print("\ncapturing FP baseline ...")
    states_fp, logits_fp = capture_layer_states(decoder, ids)

    method = RTN(bits=args.bits)
    print(f"\napplying {method.name}@{method.bits}bit ...")
    apply_quant(decoder, method)

    print("\ncapturing quantized states ...")
    states_q, logits_q = capture_layer_states(decoder, ids)

    mses = layer_mse(states_fp, states_q)
    kl = output_kl(logits_fp, logits_q)

    print(f"\n=== results: {method.name} @ {method.bits}bit ===")
    print(f"output KL : {kl:.6e}")
    print(f"per-layer MSE (x + residual at layer boundary):")
    for i, m in enumerate(mses):
        print(f"  layer {i:2d}: {m:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
