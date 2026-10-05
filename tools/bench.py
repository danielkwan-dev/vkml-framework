"""Measures vkml's prefill and decoding speed on a fixed set of models.

    python tools/bench.py --vkml-run build/release/apps/vkml-run --models <dir> [--runs 3] [--gguf] [--only <name>]

Each model processes a 512-token prompt and then generates 128 tokens, the
shapes of llama-bench's pp512 and tg128, so the figures compare with
llama.cpp's. Prompt tokens are fixed ids, not text: speed does not depend on
them. As llama-bench does, a first run warms up and is discarded, and each
figure is the mean of --runs more. With --gguf, every model runs from a GGUF
file, for comparing with llama-bench on the same files. Prints a Markdown
table.
"""

import argparse
import re
import statistics
import subprocess
from pathlib import Path

# (name, path under --models, extra vkml-run flags)
MODELS = [
    ("Qwen2.5 0.5B", "Qwen2.5-0.5B-Instruct", ["--q4"]),
    ("TinyLlama 1.1B", "TinyLlama-1.1B-Chat-v1.0", ["--q4"]),
    ("Llama 3.2 1B", "llama3.2-gguf/Llama-3.2-1B-Instruct-Q4_K_M.gguf", []),
    ("Gemma 3 1B", "gemma3-gguf/gemma-3-1b-it-Q8_0.gguf", []),
    ("Gemma 3 4B", "gemma3-4b-gguf/gemma-3-4b-it-Q4_K_M.gguf", []),
    ("Mistral 7B v0.3", "mistral-gguf/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf", []),
]

# The same models, all from GGUF files.
GGUF_MODELS = [
    ("Qwen2.5 0.5B", "qwen2.5-gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf", []),
    ("TinyLlama 1.1B", "tinyllama-gguf/tinyllama-1.1b-chat-v1.0.Q4_0.gguf", []),
    *MODELS[2:],
]

PROMPT = ",".join(str(1000 + i) for i in range(512))


def weights(path, flags):
    if flags:
        return "`" + " ".join(flags) + "`"
    match = re.search(r"(Q\d_K_M|Q\d_0)", path, re.IGNORECASE)
    return match.group(1).upper() + " GGUF" if match else "bf16"


def run(vkml_run, model, flags):
    out = subprocess.run(
        [vkml_run, "--model", model, "--context", "1024", "--tokens", PROMPT,
         "--generate", "128", "--ignore-eos", *flags],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=True).stdout
    prefill = float(re.search(r"^prefill .*\(([\d.]+) tokens/s\)", out, re.M).group(1))
    decode = float(re.search(r"^decode .*\(([\d.]+) tokens/s\)", out, re.M).group(1))
    return prefill, decode


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--vkml-run", required=True)
    parser.add_argument("--models", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--gguf", action="store_true", help="every model from a GGUF file")
    parser.add_argument("--only", help="run only the models whose name contains this")
    args = parser.parse_args()

    vkml_run = str(Path(args.vkml_run).resolve())
    print("| Model | Weights | Prefill (pp512), tokens/s | Decoding (tg128), tokens/s |")
    print("|---|---|---|---|")
    for name, path, flags in GGUF_MODELS if args.gguf else MODELS:
        if args.only and args.only not in name:
            continue
        model = args.models / path
        if not model.exists():
            print(f"| {name} | missing: {path} | | |")
            continue
        run(vkml_run, str(model), flags)  # warm-up
        results = [run(vkml_run, str(model), flags) for _ in range(args.runs)]
        prefill = statistics.mean(r[0] for r in results)
        decode = statistics.mean(r[1] for r in results)
        print(f"| {name} | {weights(path, flags)} | {prefill:.0f} | {decode:.1f} |", flush=True)


if __name__ == "__main__":
    main()
