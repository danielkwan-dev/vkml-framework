"""Checks vkml's chat templating against transformers' apply_chat_template.

    python tools/compare_chat_template.py --model <dir> --vkml-chat <path to vkml-chat>

Renders several conversations with both and compares the prompt text, and the
token ids that text encodes to. Needs transformers.
"""

import argparse
import json
import subprocess
import sys

from transformers import AutoTokenizer

CONVERSATIONS = [
    [{"role": "user", "content": "Hi"}],
    [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "What is 2+2?"}],
    [
        {"role": "user", "content": "Tell me a joke."},
        {"role": "assistant", "content": "Why did the GPU cross the road? To render the other side."},
        {"role": "user", "content": "  Another one, with trailing spaces  "},
    ],
    [{"role": "user", "content": "Multi\nline\n\nmessage with café and \U0001F642"}],
]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--vkml-chat", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    failures = 0
    for conversation in CONVERSATIONS:
        run = subprocess.run([args.vkml_chat, "--model", args.model, "--render-only"],
                             input=json.dumps(conversation), capture_output=True, text=True,
                             encoding="utf-8", check=True)
        ours = json.loads(run.stdout)
        theirs = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        # apply_chat_template adds no special tokens when it tokenizes.
        same_ids = tokenizer(ours, add_special_tokens=False).input_ids == \
            tokenizer(theirs, add_special_tokens=False).input_ids
        if ours != theirs or not same_ids:
            failures += 1
            print(f"MISMATCH for {conversation[-1]['content'][:40]!r}")
            print(f"  vkml {ours!r}")
            print(f"  HF   {theirs!r}")
    print(f"{len(CONVERSATIONS) - failures}/{len(CONVERSATIONS)} conversations render identically")
    print("PASS" if failures == 0 else "FAIL")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
