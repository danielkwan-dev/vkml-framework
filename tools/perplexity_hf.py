"""Computes a text's perplexity with HF transformers, for vkml-run --perplexity.

    python tools/perplexity_hf.py --model <dir> --text <file> [--dtype bfloat16]

Tokenizes the text, prints the ids to pass to vkml-run --tokens, and prints
the perplexity of the model, run in f32 on the CPU (or --dtype, for a model
too large for memory in f32), over those tokens. Attention is transformers'
eager implementation, the one that applies every model's details (Gemma 2's
soft cap). Needs torch and transformers.
"""

import argparse
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = parser.parse_args()

    with open(args.text, encoding="utf-8") as f:
        text = f.read().strip()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ids = tokenizer(text, return_tensors="pt").input_ids
    print("tokens", ",".join(str(i) for i in ids[0].tolist()))

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), attn_implementation="eager")
    with torch.no_grad():
        logits = model(ids).logits[0, :-1].double()
    nll = torch.nn.functional.cross_entropy(logits, ids[0, 1:]).item()
    print(f"perplexity {math.exp(nll):.4f} over {ids.shape[1] - 1} tokens "
          f"(mean negative log-likelihood {nll:.5f})")


if __name__ == "__main__":
    main()
