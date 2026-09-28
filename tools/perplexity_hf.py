"""Computes a text's perplexity with HF transformers, for vkml-run --perplexity.

    python tools/perplexity_hf.py --model <dir> --text <file>

Tokenizes the text, prints the ids to pass to vkml-run --tokens, and prints
the perplexity of the model, run in f32 on the CPU, over those tokens.
Needs torch and transformers.
"""

import argparse
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--text", required=True)
    args = parser.parse_args()

    with open(args.text, encoding="utf-8") as f:
        text = f.read().strip()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ids = tokenizer(text, return_tensors="pt").input_ids
    print("tokens", ",".join(str(i) for i in ids[0].tolist()))

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    with torch.no_grad():
        logits = model(ids).logits[0, :-1].double()
    nll = torch.nn.functional.cross_entropy(logits, ids[0, 1:]).item()
    print(f"perplexity {math.exp(nll):.4f} over {ids.shape[1] - 1} tokens "
          f"(mean negative log-likelihood {nll:.5f})")


if __name__ == "__main__":
    main()
