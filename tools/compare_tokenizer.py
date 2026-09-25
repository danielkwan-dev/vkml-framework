"""Checks vkml's tokenizer against HF tokenizers on a varied set of texts.

    python tools/compare_tokenizer.py --model <dir> --vkml-tokenize <path to vkml-tokenize>

Encodes each text with both, compares the ids, and compares decoding those
ids back. The reference is the tokenizers library applied to tokenizer.json
as written, which vkml implements. transformers' LlamaTokenizer (non-legacy
mode, the default since 4.x) replaces the file's normalizer with a Metaspace
pre-tokenizer that adds no U+2581 to text already starting with a space or
following a special token; those differences are listed but not failures.
Needs tokenizers (installed with transformers).
"""

import argparse
import json
import subprocess
import sys

from tokenizers import Tokenizer
from transformers import AutoTokenizer

TEXTS = [
    "",
    "The capital of France is",
    "Hello, world!",
    "  leading and trailing spaces  ",
    "multiple    spaces\tand\ttabs",
    "line one\nline two\n\nline four",
    "Numbers: 3.14159, 1,000,000 and 2^32 = 4294967296.",
    "Punctuation... (brackets) [square] {curly} <angle> \"quotes\" 'single'",
    "def f(x):\n    return x * 2  # a comment",
    "Accents: café, naïve, résumé, Zürich, São Paulo",
    "Chinese: 你好，世界。Japanese: こんにちは",
    "Emoji: 🙂🚀👍🏽 and symbols ∑∫√∞ ≠ ≤",
    "Special tokens in text: <s>start</s> and <unk>",
    "<s>",
    "a",
    " ",
    "\n",
    "The quick brown fox jumps over the lazy dog. " * 20,
    "Supercalifragilisticexpialidocious antidisestablishmentarianism",
    "URLs: https://example.com/path?query=1&x=y#frag and emails a.b@c.org",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--vkml-tokenize", required=True)
    args = parser.parse_args()

    reference = Tokenizer.from_file(f"{args.model}/tokenizer.json")
    transformers_tok = AutoTokenizer.from_pretrained(args.model)
    run = subprocess.run(
        [args.vkml_tokenize, f"{args.model}/tokenizer.json"],
        input="".join(json.dumps(t) + "\n" for t in TEXTS),
        capture_output=True, text=True, encoding="utf-8", check=True)
    results = [json.loads(line) for line in run.stdout.splitlines()]

    failures = 0
    differ_from_transformers = []
    for text, ours in zip(TEXTS, results, strict=True):
        want_ids = reference.encode(text).ids
        want_text = reference.decode(want_ids, skip_special_tokens=True)
        if transformers_tok(text).input_ids != want_ids:
            differ_from_transformers.append(text)
        ids_ok = ours["ids"] == want_ids
        text_ok = ours["decoded"] == want_text
        if not (ids_ok and text_ok):
            failures += 1
            print(f"MISMATCH {text[:60]!r}")
            if not ids_ok:
                print(f"  vkml ids {ours['ids'][:24]}")
                print(f"  ref  ids {want_ids[:24]}")
            if not text_ok:
                print(f"  vkml decoded {ours['decoded'][:60]!r}")
                print(f"  ref  decoded {want_text[:60]!r}")
    print(f"{len(TEXTS) - failures}/{len(TEXTS)} texts match tokenizer.json")
    for text in differ_from_transformers:
        print(f"note: transformers tokenizes {text[:40]!r} differently (non-legacy Metaspace)")
    print("PASS" if failures == 0 else "FAIL")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
