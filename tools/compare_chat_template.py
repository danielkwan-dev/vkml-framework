"""Checks vkml's chat templating against transformers' apply_chat_template.

    python tools/compare_chat_template.py --model <dir> --vkml-chat <path to vkml-chat>

Renders several conversations with both, two of them with tools, a tool call
and its result, and compares the prompt text and the token ids that text
encodes to. Needs transformers.
"""

import argparse
import json
import subprocess
import sys

from jinja2.exceptions import TemplateError
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


# Conversations with tools, as OpenAI's API sends them to vkml-server
# (arguments as a JSON string; apply_chat_template takes them as an object).
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather in a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]
CALL = {"id": "call00010", "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}
CONVERSATIONS += [
    {"messages": [{"role": "user", "content": "What's the weather in Paris?"}], "tools": TOOLS},
    {"messages": [{"role": "user", "content": "What's the weather in Paris?"},
                  {"role": "assistant", "content": None, "tool_calls": [CALL]},
                  {"role": "tool", "tool_call_id": "call00010", "content": "Sunny, 24 C"}],
     "tools": TOOLS},
]


def for_hf(conversation):
    """apply_chat_template's messages and tools for a conversation."""
    if isinstance(conversation, list):
        return conversation, None
    messages = json.loads(json.dumps(conversation["messages"]))
    for m in messages:
        for call in m.get("tool_calls", []):
            call["function"]["arguments"] = json.loads(call["function"]["arguments"])
    return messages, conversation["tools"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--vkml-chat", required=True)
    parser.add_argument("--vkml-source", help="what vkml reads instead of the model "
                        "directory, such as a GGUF file of the same model")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    failures = skipped = 0
    for conversation in CONVERSATIONS:
        run = subprocess.run([args.vkml_chat, "--model", args.vkml_source or args.model,
                          "--render-only"],
                             input=json.dumps(conversation), capture_output=True, text=True,
                             encoding="utf-8")
        try:
            messages, tools = for_hf(conversation)
            theirs = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False,
                                                   add_generation_prompt=True)
        except TemplateError as e:
            # A template may refuse a conversation (Gemma 2's, a system
            # message): vkml must refuse it too.
            if run.returncode == 0:
                failures += 1
                print(f"MISMATCH: HF raises {e}, but vkml renders {run.stdout.strip()!r}")
            continue
        except TypeError as e:
            # A template without tool calls adds an assistant's null content
            # to a string: HF's Jinja fails where vkml's renders it empty.
            print(f"skipped: HF's template cannot render it ({e})")
            skipped += 1
            continue
        if run.returncode != 0:
            failures += 1
            print(f"MISMATCH: vkml fails ({run.stderr.strip()}) where HF renders {theirs!r}")
            continue
        ours = json.loads(run.stdout)
        # apply_chat_template adds no special tokens when it tokenizes.
        same_ids = tokenizer(ours, add_special_tokens=False).input_ids == \
            tokenizer(theirs, add_special_tokens=False).input_ids
        if ours != theirs or not same_ids:
            failures += 1
            print(f"MISMATCH for {str(for_hf(conversation)[0][-1]['content'])[:40]!r}")
            print(f"  vkml {ours!r}")
            print(f"  HF   {theirs!r}")
    compared = len(CONVERSATIONS) - skipped
    print(f"{compared - failures}/{compared} conversations render identically"
          + (f" ({skipped} skipped)" if skipped else ""))
    print("PASS" if failures == 0 else "FAIL")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
