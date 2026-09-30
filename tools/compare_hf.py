"""Checks vkml-run against HF transformers on the same checkpoint and tokens.

    python tools/compare_hf.py --model <dir> --tokens 1,450,7483 \
        --logits <file from vkml-run --dump-logits> [--generated 3681,29889,...]

Runs the model in f32 on the CPU (or --dtype bfloat16, for a model too large
for memory in f32), with transformers' eager attention, compares the
last-token logits with vkml's, and, given vkml's greedy continuation, checks
HF generates the same tokens.
Needs torch and transformers.
"""

import argparse
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def ids(text):
    return [int(t) for t in text.split(",") if t]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokens", required=True, type=ids)
    parser.add_argument("--logits", required=True, help="raw f32 logits from vkml-run")
    parser.add_argument("--generated", type=ids, default=[])
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), attn_implementation="eager")
    model.eval()

    with torch.no_grad():
        hf = model(torch.tensor([args.tokens])).logits[0, -1].double().numpy()
    ours = np.fromfile(args.logits, dtype="<f4").astype(np.float64)
    if ours.shape != hf.shape:
        sys.exit(f"vkml wrote {ours.size} logits, HF has {hf.size}")

    diff = np.abs(ours - hf)
    scale = np.abs(hf).max()
    print(f"prompt     {tokenizer.decode(args.tokens)!r}")
    print(f"logits     max |diff| {diff.max():.3e}, relative to max |logit| {diff.max() / scale:.3e}")
    print(f"argmax     vkml {ours.argmax()}, HF {hf.argmax()}")
    top_hf = set(np.argsort(-hf)[:10])
    top_ours = set(np.argsort(-ours)[:10])
    print(f"top-10     {len(top_hf & top_ours)}/10 shared")
    ok = ours.argmax() == hf.argmax() and diff.max() / scale < 1e-3

    if args.generated:
        with torch.no_grad():
            # Plain greedy decoding, overriding what generation_config.json
            # asks for (Qwen2.5's asks for a repetition penalty).
            out = model.generate(torch.tensor([args.tokens]), max_new_tokens=len(args.generated),
                                 do_sample=False, repetition_penalty=1.0,
                                 no_repeat_ngram_size=0)
        hf_generated = out[0, len(args.tokens):].tolist()
        same = hf_generated == args.generated
        print(f"greedy     {'same' if same else 'DIFFERENT'} {len(args.generated)} tokens")
        print(f"  vkml     {tokenizer.decode(args.generated)!r}")
        print(f"  HF       {tokenizer.decode(hf_generated)!r}")
        ok = ok and same

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
