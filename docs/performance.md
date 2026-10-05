# Performance

All figures are from an Intel Iris Xe laptop GPU (integrated, shared LPDDR4x
memory), plugged in. Integrated GPUs change clocks a lot between runs, by up
to 50%: figures are the best of several runs, and differences under 15% are
noise.

## Benchmark

`tools/bench.py` runs each model on a 512-token prompt, then generates 128
tokens (llama-bench's pp512 and tg128). A first run warms up and is
discarded; figures are the mean of 3 more. Measured 2026-10-05, plugged in,
from GGUF files (`--gguf`):

| Model | Weights | Prefill, tokens/s | Decoding, tokens/s |
|---|---|---|---|
| Qwen2.5 0.5B | Q4_K_M | 971 | 62.7 |
| TinyLlama 1.1B | Q4_0 | 457 | 51.2 |
| Llama 3.2 1B | Q4_K_M | 425 | 39.6 |
| Gemma 3 1B | Q8_0 | 634 | 32.3 |
| Gemma 3 4B | Q4_K_M | 153 | 13.4 |
| Mistral 7B v0.3 | Q4_K_M | 76 | 8.8 |

From Hugging Face checkpoints quantized as they load, best of 3: Qwen2.5 0.5B
`--q4` 1,202 and 71.3, TinyLlama 1.1B `--q4` 469 and 50.6.

The sections below were measured over the project's development, often as
best of several runs; they show what each change gained.

## Against llama.cpp

llama.cpp's Vulkan backend (Windows release b11430) on the same GGUF files,
in the same session: `llama-bench -p 512 -n 128 -ngl 99 -r 3`, which also
warms up and reports the mean. It detects the same device features vkml
uses (`int dot: 1`, no matrix cores).

| Model | Prefill, vkml | llama.cpp | Decoding, vkml | llama.cpp |
|---|---|---|---|---|
| Qwen2.5 0.5B Q4_K_M | 971 | 1238 ± 7 | 62.7 | 60.9 ± 0.5 |
| TinyLlama 1.1B Q4_0 | 457 | 709 ± 1 | 51.2 | 42.5 ± 0.2 |
| Llama 3.2 1B Q4_K_M | 425 | 576 ± 1 | 39.6 | 34.8 ± 0.5 |
| Gemma 3 1B Q8_0 | 634 | 522 ± 2 | 32.3 | 22.7 ± 0.1 |
| Gemma 3 4B Q4_K_M | 153 | 172 ± 1 | 13.4 | 11.8 ± 0.0 |
| Mistral 7B v0.3 Q4_K_M | 76 | 86 ± 0 | 8.8 | 7.4 ± 0.0 |

vkml decodes 3-42% faster, though it reads more bytes for some layers
(q6_K runs as Q8_0). llama.cpp's prompt processing is 11-55% faster on five
of the six. Differences under 15% are within this GPU's run-to-run noise. This
is one integrated GPU: llama.cpp is tuned across many, and the comparison may
come out differently elsewhere.

## Decoding

| Model | Load | bf16 | `--q8` | `--q4` |
|---|---|---|---|---|
| TinyLlama 1.1B | 2–5 s | ~22 tokens/s | ~34 tokens/s | ~49 tokens/s |
| SmolLM2 360M | 1–1.5 s | ~47 tokens/s | ~65 tokens/s | ~76 tokens/s |
| Qwen2.5 0.5B | 2 s | ~39 tokens/s | ~55 tokens/s | ~67 tokens/s |
| Qwen3 0.6B | 1.5 s | ~32 tokens/s | ~45 tokens/s | ~55 tokens/s |
| Llama 3.2 1B | 3 s | ~18 tokens/s | ~29 tokens/s | ~34 tokens/s |
| Gemma 3 1B | 2–3 s | ~18 tokens/s | ~26 tokens/s | ~35 tokens/s |
| Gemma 2 2B | 20 s | ~7 tokens/s | ~13 tokens/s | ~13 tokens/s |

GGUF files:

| File | Load | Decoding |
|---|---|---|
| Llama 3.2 1B Q4_K_M | 3 s | ~33–40 tokens/s |
| Gemma 3 1B Q8_0 | | ~34 tokens/s |
| Gemma 3 4B Q4_K_M | ~11 s | ~10.5–12.6 tokens/s |
| Mistral 7B Instruct v0.3 Q4_K_M (4.4 GB) | ~15 s | ~8.4 tokens/s |

### Memory bandwidth is the limit

Each generated token reads every weight once. For TinyLlama in bf16 that is
2.07 GB per token, which the matrix-vector kernels stream at about 51 GB/s
(40.6 ms of GPU time per token): the most this machine sustains for large
reads from uncached memory. So the way to decode faster is to read fewer
bytes.

### int8 dot products

`--q8` stores each block of 32 weights as 32 bytes plus a scale (1.1 GB per
token for TinyLlama), and `--q4` as 16 bytes plus a scale (0.6 GB). Unpacking
them to floats as they were read, the matrix-vector kernels could not issue
loads fast enough to beat the 16-bit rate by much.

On GPUs with accelerated int8 dot products (`vkml-info` says), vkml instead
quantizes the activations to int8 too, once per layer input, and multiplies
int8 by int8 four at a time, each lane loading 16 bytes of both operands.
**That took TinyLlama from 28 to 49 tokens/s with `--q4`**, and moves
perplexity by under 0.5% (see [correctness.md](correctness.md)).
`ContextOptions::integer_dot_product = false` (or `--no-dot`) turns it off.

### f16 scales

Scales were f32; f16 ones save 5.5% of q8_0's bytes, 10% of q4_0's and 17% of
q4_1's. On Gemma 3 4B's shapes (`vkml-bench --decode`) they took q8_0 from
86–92 ms per token to 82–84, q4_1 from 65 to 60, and q4_0 from 57 to 56. They
move Gemma 3 4B's perplexity by under 0.4%.

Llama 3.2 1B's Q4_K_M file generates at ~33 tokens/s with its q4_K layers
run as Q4_1, against ~29 widened to Q8_0.

### Argmax on the GPU

Greedy decoding finds the next token on the GPU and reads back its 4 bytes,
not every logit: copying and scanning Gemma's 262,144 logits took ~4.5 ms of
host time per token while the GPU sat idle. Gemma 3 1B's Q8_0 file went from
~29 to ~34 tokens/s, Llama 3.2 1B's Q4_K_M from ~36 to ~40.

### Overlapping recording and execution

The model submits each layer as soon as it is recorded, so the GPU runs one
layer while the host records the next. That took TinyLlama from 18.6 to 22
tokens/s.

### Long contexts

Attention for a generated token was three matrix products with one query
row each, far too little parallel work for a long cache. It is now one kernel
split over chunks of 128 keys, then a small one merging the chunks.

| After | Model | Before | After | Empty cache |
|---|---|---|---|---|
| 1,500 tokens | TinyLlama bf16 | | ~20 tokens/s | ~22 tokens/s |
| 1,500 tokens | TinyLlama `--q4` | ~25 tokens/s | ~40 tokens/s | ~49 tokens/s |
| ~2,000 tokens | Gemma 3 1B Q8_0 | ~23 tokens/s | ~30 tokens/s | |
| ~2,000 tokens | Gemma 3 4B Q4_K_M | ~9.3 tokens/s | ~12.6 tokens/s | |

With `--kv-f16`, the cache takes half the memory for about the same speed:
the decoding attention kernel reads half the bytes and widens them with the
hardware's conversion, taking 4-5 ms per token at 1,500 tokens either way.
SmolLM2's wikitext perplexity moves from 7.6559 to 7.6527 with `--q8`.

## Prefill

Prompts of up to 8 rows take the matrix-vector path. Longer ones use a tiled
kernel, at about 390 GFLOP/s in f32. With quantized weights they take a tiled
int8 kernel, whose staged tiles hold four values per word and whose dot
products do four multiply-adds each. The f32 tiled kernel is held back by
loading its tiles and by its multiply-adds about equally, so this helps both:

| Prompt | bf16 | `--q8` | `--q4` |
|---|---|---|---|
| TinyLlama 1.1B, 106 tokens | ~120 tokens/s | | |
| TinyLlama 1.1B, 512 tokens | ~165 tokens/s | ~345 tokens/s | ~460 tokens/s |
| Qwen2.5 0.5B, 222 tokens | ~415 tokens/s | ~1030 tokens/s | ~920 tokens/s |
| Llama 3.2 1B Q4_K_M GGUF, 707 tokens | | | ~400 tokens/s |

After those prompts the logits stay within 2% (TinyLlama) and 5% (Qwen2.5) of
the largest one against `transformers` with `--q8`, the top ten tokens the
same. The int8 activations account for about 40% of that, and under 1% of
perplexity.

A long prompt's attention scores would be large (288 MB per layer for 1,500
tokens of TinyLlama), and allocating them every layer stalled the GPU for most
of the prompt. Queries now go in chunks whose scores stay under 64 MB, each
against only the keys it can see: a 1,500-token prompt with `--q4` takes
5.3 s instead of 13.

## What did not help

Kernel tuning on this GPU is done. Staging more K blocks, `uvec4` shared
tiles, larger register tiles and whole-block gemv lanes were all slower.

## Profiling

`vkml-run --profile` times every kernel with GPU timestamps:

```
generation: GPU busy 42.99 ms per step, wall clock 53.74 ms per step
  kernel          calls     total ms      ms/step   share
  gemv            11328      2601.30       40.645   94.5%
  rms_norm         2880        35.46        0.554    1.3%
  matmul           1408        29.62        0.463    1.1%
  ...
```

`vkml-bench` times the matmul shapes of a LLaMA forward pass, checks every
result, and cycles through enough copies of each weight matrix that caches
cannot flatter the bandwidth.

`vkml-bench --decode <model>` times one decoding step's matrix-vector
products for a real model's shapes (only its config is read) in q8_0, q4_0
and q4_1. It runs four rounds, each timing every format, and keeps each
matrix's best time, because the GPU's clocks shift for tens of seconds at a
time. Across three runs for Gemma 3 4B, the per-token sums of those bests
agree to within 5%, while medians move by 10–20%. `--f32-scales` times f32
scales instead.
