# tools

Checks against Hugging Face, test-data generators, and build helpers. The
Python scripts need `transformers` (and `torch` where noted); each one's
docstring has the details. On Windows, run them with `PYTHONIOENCODING=utf-8`.

## Checks against transformers

| Script | What it checks | Run |
|---|---|---|
| `compare_hf.py` | last-token logits and greedy generation vs `transformers` (torch) | `python tools/compare_hf.py --model <dir> --tokens <ids> --logits <vkml-run --dump-logits file> [--generated <ids>]` |
| `perplexity_hf.py` | a text's perplexity in `transformers`, and the ids to give `vkml-run --tokens` (torch) | `python tools/perplexity_hf.py --model <dir> --text <file> [--dtype bfloat16]` |
| `compare_tokenizer.py` | token ids and decoded text vs `tokenizers` on 26 texts | `python tools/compare_tokenizer.py --model <dir> --vkml-tokenize <exe>` |
| `compare_chat_template.py` | prompts (and their ids) vs `apply_chat_template`, tools included | `python tools/compare_chat_template.py --model <dir> --vkml-chat <exe>` |
| `slice_checkpoint.py` | copies a checkpoint's first layers, so large models fit `transformers` (torch) | `python tools/slice_checkpoint.py --model <dir> --layers 4 --out <dir>` |

## Generators

| Script | Writes |
|---|---|
| `gen_jinja_cases.py` | `tests/data/jinja_cases.json`: templates and real Jinja's output for them |
| `gen_kquant_cases.py` | `tests/data/kquant_cases.json`: random K-quant blocks and gguf-py's decoding of them |
| `gen_unicode_tables.py` | `src/io/unicode_tables.inc`: letter/number ranges and NFC data |

Each writes to standard output: `python tools/gen_jinja_cases.py > tests/data/jinja_cases.json`.

## Benchmark

`bench.py` measures prefill (512 tokens) and decoding (128 tokens) speed on a
fixed set of models, best of 3, as a Markdown table:
`python tools/bench.py --vkml-run build/release/apps/vkml-run --models <dir>`.

## Build helpers

- `install-deps-ubuntu.sh`: compiler, CMake, Ninja, glslc, the Vulkan loader
  and lavapipe on Ubuntu 24.04.
- `ci.Dockerfile`: CI's environment, to reproduce it locally.
- `consumer/`: a project that uses vkml through `add_subdirectory`, built in
  CI to check vkml works as a dependency.
