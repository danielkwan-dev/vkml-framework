# Correctness

vkml is checked at three levels: unit tests against double-precision
references, a tiny random model against a from-scratch reference
implementation, and real checkpoints against Hugging Face `transformers`.

## Unit tests

- 220 Catch2 tests run under the Vulkan validation layers, and each one
  asserts that the layers reported no errors. CI runs them on Mesa's lavapipe.
- Operators are compared with double-precision references on the host. Matmul
  uses the standard rounding bound for f32 dot products as its tolerance, so
  tests do not pass or fail by luck.
- Tests that guard subtle behaviour (deferred frees, rope scaling, rope
  layout, and more) were checked to fail when that behaviour is broken.
- **A tiny random model** is checked against a from-scratch double-precision
  implementation inside the suite: prefill, token-by-token decoding through
  the KV cache, grouped-query attention, tied embeddings, bf16 weights, LLaMA
  3.1 rope scaling, Qwen2's biases, Qwen3's q and k norms, sliding windows in
  every layer or some, and Gemma 2's and 3's layouts (Gemma 2's soft caps set
  low enough to bend most scores and logits). It is written both as a Hugging
  Face directory and as a GGUF file.
- **vkml-server** is tested over HTTP, on a random tiny model and on a
  "script model" (no layers, one-hot embeddings, an output projection mapping
  each token to the next) that answers a fixed text. That makes tool calls,
  `<think>` blocks and streaming testable end to end. It found a 500 on replies
  cut inside a UTF-8 character.
- 230 corrupted GGUF headers load without a crash.

## Against transformers

The scripts are in `tools/` (see [tools/README.md](../tools/README.md)).

### Logits and greedy generation (`compare_hf.py`)

Last-token logits agree with `transformers` to within a few parts per million (~1e-6)
of the largest logit, and greedy generation matches token for token:
TinyLlama 1.1B, SmolLM2 360M, Qwen2.5 0.5B, Qwen3 0.6B, Llama 3.2 1B (with
its LLaMA 3.1 rope scaling), and `transformers`' own tiny random Mistral,
converted to safetensors (with its sliding window set to 16 tokens, a
61-token prompt's logits still agree to 3e-7). Gemma 3 1B agrees to 1.3e-6
after a short prompt, and 32 greedy tokens match.

Models too large for `transformers` in f32 on this machine are checked on a
copy of their first few layers made by `slice_checkpoint.py`:

| Model (slice) | Max logit difference (of the largest) | Also |
|---|---|---|
| OLMo 2 1B (4 layers) | 1.3e-6 | 8 greedy tokens the same |
| Granite 3.3 2B | 2.5e-7 | same perplexity to 6 digits |
| SmolLM3 3B (rope skipped in its fourth layer) | 6.4e-7 | same perplexity to 6 digits |
| Phi-4-mini | 5.8e-7 | once `transformers` has the Q8_0-rounded embeddings vkml uses (too large for a GPU buffer), with `--no-dot` |

### Tokenizers (`compare_tokenizer.py`)

Identical ids and decoded text to the `tokenizers` library on 26 texts
(scripts, emoji, digits, whitespace, special tokens, decomposed accents) and
on 200,000-character texts, for SentencePiece (old-style and Metaspace, as
Mistral 7B v0.3), byte-level, LLaMA 3-, Qwen2- and o200k-style (Phi-4)
tokenizers, from `tokenizer.json` or a GGUF file. ggml-org's Gemma 3 1B GGUF
file, whose tokenizer has only scores, gets merges identical to the HF
tokenizer's 514,906, in order.

NFC normalization agrees with Python's `unicodedata` on every code point,
alone and decomposed, and on 20,000 random sequences of marks (checked once
while writing it; the unit tests keep the tricky cases).

### Chat templates (`compare_chat_template.py`)

Identical prompts to `apply_chat_template` on twelve models' templates,
including tool-calling branches, and the templates inside GGUF files
(TinyLlama, Qwen2.5, Qwen3, Gemma 3). The Jinja interpreter's own expected
outputs come from real Jinja with `transformers`' settings
(`gen_jinja_cases.py`).

### GGUF decoding

The K-quant and older block decoders match gguf-py, llama.cpp's Python
package, on random blocks (`gen_kquant_cases.py`). GGUF tokenizers and chat
templates match the HF ones on the 26 texts and 4 conversations, for every
GGUF model listed below.

### Log probabilities

vkml-server's `logprobs` for Qwen2.5 0.5B match `transformers` to four
decimals (within 6.5e-5).

### lm-evaluation-harness

On the first 100 ARC-Easy questions, Qwen2.5 0.5B scores the same through
vkml-server as through the harness's `transformers` backend: accuracy 0.58,
normalized 0.61.

## Perplexity

`vkml-run --perplexity` scores a text from the logits at every position, in
chunks; `tools/perplexity_hf.py` does the same with `transformers`. On a
250-token passage the two agree to four decimals.

Compare only on the token ids `transformers` saw (`vkml-run --tokens`): one
extra token moved Gemma 2's figure by 5%.

The int8 activations vkml uses with quantized weights (see
[performance.md](performance.md)) change summation order, which moves
perplexity by up to ~0.4%. `--no-dot` keeps activations in f32 and gives the
same figures as without quantization (Gemma 3 1B 20.6654 and Qwen3 0.6B
13.0547 either way).

### A 250-token passage

| Model | transformers (f32) | vkml | vkml `--q8` | vkml `--q4` |
|---|---|---|---|---|
| TinyLlama 1.1B | 6.8482 | 6.8482 | 6.8244 | 6.8275 |
| SmolLM2 360M | 7.0729 | 7.0729 | 7.0882 | 8.3021 |

### wikitext-2 test set

| Model | Tokens | transformers | vkml | `--q8` | GGUF Q8_0 | Other |
|---|---|---|---|---|---|---|
| SmolLM2 360M | 1,636 | 7.6412 | 7.6412 | 7.6437 | | `--q4` 8.68 (9.83 with a 4-bit output projection) |
| Qwen3 0.6B | 1,467 | 11.1489 | 11.1489 | 11.1378 | 11.1101 | `--q4` 12.38 |
| Gemma 3 1B (past its 512-token window) | 1,428 | 14.9329 | 14.9329 | 14.9504 | 14.9346 | |
| Gemma 3 1B (past the window's ring buffer) | 2,057 | 18.3057 | 18.3057 | | | |
| Gemma 2 2B | 718 | 8.7370 (bf16) | 8.7274 | | 8.7362 | embeddings quantized to Q8_0 (1 GB buffer) |
| OLMo 2 1B | 3,000 chars | 9.1129 (bf16) | 9.1048 | | 9.1131 | |
| Granite 3.3 2B | 915 | | | 5.53 | 5.54 | |
| SmolLM3 3B | 704 | | | 7.07 | 7.06 | |
| Phi-4-mini | 708 | | | 7.10 | 7.11 | |

Small models lose the most to 4 bits.

### GGUF files and K-quants

On the 250-token passage:

| Model | HF checkpoint | `--q4` | GGUF Q8_0 | GGUF Q4_K_M | Other |
|---|---|---|---|---|---|
| Qwen2.5 0.5B | 9.45 | 10.71 | 9.65 | 10.08 | |
| Llama 3.2 1B | 8.94 | 10.16 | 8.94 | 9.31 | Q3_K_L 10.09 |

Against `transformers` in f32, TheBloke's TinyLlama Q8_0 file gives logits
within 1.6% of the largest, and its greedy generation is identical to `--q8`
on the HF checkpoint. Llama 3.2 1B's Q8_0 file is within 1.1%.

On the first 3,000 characters of the wikitext passage:

| Model (GGUF) | Perplexity |
|---|---|
| Gemma 3 1B Q8_0 | 20.53 |
| Gemma 3 4B Q4_K_M | 12.58 |
| Mistral 7B Instruct v0.3 Q4_K_M | 4.54 |

Gemma 3 4B and Mistral 7B are too large for `transformers` in this machine's
16 GB; only their tokenizers and chat templates were checked against it.
Running q4_K as Q4_1 with f16 scales moves Gemma 3 4B's perplexity by under
0.4% against exact f32 scales (12.62 and 12.67, measured before decoding
attention's summation order changed).

## Known differences

- Phi-3/Phi-4's LongRoPE takes its long factors for every position when
  `--context` is past the original 4,096 positions, as llama.cpp does;
  `transformers` switches only once a sequence is that long, so shorter ones
  then differ slightly.
- Granite 3.3's GGUF chat template spaces the system prompt differently from
  the HF one; vkml renders each faithfully.
