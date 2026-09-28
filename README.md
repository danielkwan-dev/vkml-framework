# vkml

A GPU tensor library and LLM inference engine written from scratch in C++20 on
**Vulkan compute**, so it runs on GPUs from any vendor with a Vulkan 1.2
driver, not only CUDA hardware. It is developed on an Intel laptop GPU, and CI
runs it on Mesa's lavapipe, a CPU implementation of Vulkan.

It loads Hugging Face checkpoints of LLaMA-architecture models (and Qwen2,
which adds attention biases), tokenizes and
formats chat prompts the way `transformers` does, and generates text on the
GPU:

```
$ vkml-chat --model models/SmolLM2-360M-Instruct --temperature 0
SmolLM2-360M-Instruct on Intel(R) Iris(R) Xe Graphics. /reset starts over, /quit leaves.

> What is the capital of France?
The capital of France is Paris.
  [7 tokens, 14.9 tokens/s]

> And what is its population, roughly?
The population of Paris is approximately 2.2 million people.
  [13 tokens, 22.5 tokens/s]
```

That conversation is token-for-token what Hugging Face `transformers` generates
greedily for the same messages (see [Correctness](#correctness)).

**Models run end to end and checked against `transformers`:** TinyLlama 1.1B
(Chat), SmolLM2 360M (base and Instruct). The pieces LLaMA 3.x adds (its
tokenizer's pre-tokenization, rope scaling, grouped-query attention) are tested
individually, but no LLaMA 3 checkpoint has been run yet; its weights need
Meta's license accepted on Hugging Face.

## Features

- **Tensors on the GPU**: f32, f16, bf16 and i32; asynchronous execution with
  automatic lifetime management.
- **Operators**: elementwise arithmetic, SiLU/GeLU, softmax, RMSNorm, matmul
  (batched, transposed, 16-bit weights), permute/transpose, embedding lookup,
  rotary position embeddings (including LLaMA 3.1 scaling), and scaled
  dot-product attention with causal masking, grouped-query attention and a KV
  cache.
- **Models**: `vkml::Llama` runs LLaMA-architecture models (LLaMA, TinyLlama,
  SmolLM) and Qwen2/Qwen2.5. It loads `config.json` and sharded `.safetensors`,
  keeps weights in their checkpoint precision (bf16/f16) or quantizes them to
  8 or 4 bits (Q8_0, Q4_0) on loading, and decodes through a KV cache.
- **Tokenizers**: SentencePiece-style BPE (LLaMA 1/2, TinyLlama) and
  byte-level BPE with the GPT-2, LLaMA 3 and Qwen2 pre-tokenization rules
  (SmolLM, LLaMA 3, Qwen2, GPT-2 family), with NFC normalization, read from
  `tokenizer.json`. Files using anything else
  are rejected with an error naming it.
- **Chat templates**: a small Jinja interpreter runs the template each model
  ships in `tokenizer_config.json`, exactly as `apply_chat_template` does.
- **Sampling**: temperature, top-k and top-p, seeded.
- **Command-line tools**: `vkml-chat` (interactive chat), `vkml-run` (prompt
  completion), `vkml-info` (device report), `vkml-bench` (matmul benchmark),
  `vkml-tokenize`.

## Building

Requirements: CMake 3.25+, Ninja, a C++20 compiler, `glslc` (from shaderc or
the Vulkan SDK) and a Vulkan 1.2 driver. All C++ dependencies (Vulkan headers,
volk, vk-bootstrap, VMA, nlohmann/json, Catch2) are fetched and pinned at
configure time.

**Linux** (Ubuntu 24.04):

```sh
bash tools/install-deps-ubuntu.sh   # compiler, CMake, Ninja, glslc, Vulkan loader + lavapipe
cmake --preset release
cmake --build --preset release
```

**Windows** (in an MSYS2 UCRT64 shell; the GPU driver provides Vulkan):

```sh
pacman -S mingw-w64-ucrt-x86_64-{gcc,cmake,ninja,shaderc,vulkan-validation-layers}
cmake --preset release -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc
cmake --build --preset release
```

Presets: `debug` (validation layers on), `release`, and `ci` (debug with
warnings as errors). Binaries land in `build/<preset>/apps/`; MinGW builds link
their runtime statically, so they run outside the MSYS2 shell too.

Tests: `cmake --preset debug && cmake --build --preset debug && ctest --preset
debug`. They need a Vulkan device; without a GPU, Mesa's lavapipe works (it is
what CI uses). On Windows the loader finds MSYS2's validation layers only with
`VK_ADD_LAYER_PATH=C:\msys64\ucrt64\bin` set; without them the validation tests
skip.

## Usage

Download a checkpoint in Hugging Face format (`config.json`, `*.safetensors`,
`tokenizer.json`, `tokenizer_config.json`), for example with
`huggingface_hub.snapshot_download("HuggingFaceTB/SmolLM2-360M-Instruct",
local_dir="models/SmolLM2-360M-Instruct")`. Then:

```sh
vkml-chat --model models/SmolLM2-360M-Instruct            # chat; /reset, /quit
vkml-run  --model models/TinyLlama-1.1B-Chat-v1.0 --prompt "The capital of France is"
vkml-info                                                   # the GPU vkml picked
```

`vkml-chat` options: `--system`, `--temperature`, `--top-k`, `--top-p`,
`--seed`, `--max-reply`, `--context`, `--q8` or `--q4`. `VKML_DEVICE=<name>` or `--device
<name>` picks a GPU by (part of) its name.

As a library ([examples/quickstart.cpp](examples/quickstart.cpp)):

```cpp
#include <vkml/vkml.hpp>

vkml::Context context;  // the GPU, its queue and memory

auto a = vkml::Tensor::from_data<float>(context, std::vector<float>{1, 2, 3, 4}, {2, 2});
auto b = vkml::Tensor::from_data<float>(context, std::vector<float>{5, 6, 7, 8}, {2, 2});
vkml::Tensor c = vkml::softmax(vkml::matmul(a, b));  // recorded for the GPU
std::vector<float> values = c.to_vector<float>();      // waits and reads back

const std::string dir = "models/SmolLM2-360M-Instruct";
vkml::Llama model = vkml::Llama::load(context, dir, /*context_length=*/1024);
vkml::Tokenizer tokenizer{dir + "/tokenizer.json"};
vkml::ChatTemplate chat = vkml::ChatTemplate::load(dir);

std::vector<vkml::ChatMessage> messages{{"user", "What is the capital of France?"}};
std::string prompt = chat.render(messages, /*add_generation_prompt=*/true);
std::vector<float> logits = model.forward(tokenizer.encode(prompt, /*add_bos=*/false)).to_vector<float>();
// ...sample a token with vkml::Sampler, forward it, repeat: see the example.
```

## Design

```
include/vkml/   public API: Context, Tensor, ops, Llama, Tokenizer, ChatTemplate, Sampler
src/hal/        Vulkan: device selection, buffers (VMA), compute pipelines, command stream
src/core/       runtime (kernel cache, deferred frees), tensors
src/ops/        operator host code, one file per family
src/io/         safetensors, tokenizers, Unicode tables, Jinja, sampling
src/models/     the LLaMA model
shaders/        GLSL compute kernels, compiled to SPIR-V and embedded at build time
```

Some decisions worth knowing:

- **No Vulkan in public headers.** The API is plain C++; Vulkan stays behind
  `src/hal`, and entry points load at run time through volk, so a machine
  without Vulkan fails with a clear error rather than at startup.
- **Asynchronous by default.** Operators record work into one command stream
  and return at once; only reading a tensor waits. Buffers freed while the GPU
  may still use them are retired against the stream's timeline semaphore and
  returned to a pool when that work completes, so a forward pass reuses the
  same buffers every token instead of creating hundreds (which gets slower the
  more allocations a process holds).
- **The GPU starts before the host finishes.** The stream rotates through
  several command buffers, and the model submits each layer as soon as it is
  recorded, so recording the next layer overlaps running the last one. This
  alone took TinyLlama from 18.6 to 22 tokens/s.
- **Kernels are tuned to the device, not hard-coded.** Workgroup sizes and
  tiles come from the device's limits through specialization constants; the
  Vulkan spec only guarantees 128 invocations and 16 KiB of shared memory.
- **16-bit weights without 16-bit hardware features.** Kernels read f16/bf16
  weights as packed 32-bit words and widen them in registers, so memory use and
  bandwidth halve on any GPU while arithmetic stays f32. f16 decoding is done by
  hand so subnormals survive.
- **Rotary embedding tables are computed on the host in double precision.**
  Vulkan only bounds `sin`/`cos` error within [-π, π], and positions far into a
  context make angles of thousands of radians.
- **Decoding and prefill use different matmul kernels.** A matrix-vector kernel
  reads the weights once at close to memory bandwidth for the few-row products
  of decoding; a register-blocked tiled kernel, with its block shape chosen by
  row count, handles prompts.

## Correctness

- **Unit tests** (135, Catch2) run under the Vulkan validation layers, and each
  asserts the layers reported no errors. Operators are compared against
  double-precision references on the host; matmul uses the standard rounding
  bound for f32 dot products as its tolerance, so tests do not pass or fail by
  luck. Tests that guard subtle behaviour (deferred frees, rope scaling, rope
  layout) have been checked to fail when that behaviour is broken.
- **A tiny random LLaMA** is checked against a from-scratch double-precision
  implementation inside the test suite: prefill, token-by-token decoding
  through the KV cache, grouped-query attention, tied embeddings, bf16 weights
  and LLaMA 3.1 rope scaling.
- **Against Hugging Face**, with the scripts in `tools/`:
  - `compare_hf.py`: last-token logits agree with `transformers` to within a
    few parts per million of the largest logit, and greedy generation matches
    token for token (TinyLlama 1.1B, SmolLM2 360M, Qwen2.5 0.5B).
  - `compare_tokenizer.py`: identical ids and decoded text to the `tokenizers`
    library on 26 texts (scripts, emoji, digits, whitespace, special tokens,
    decomposed accents) for SentencePiece, byte-level, LLaMA 3- and
    Qwen2-style tokenizers.
  - NFC normalization agrees with Python's `unicodedata` on every code point,
    alone and in its decomposed forms, and on 20,000 random sequences of marks
    (checked once while writing it; the unit tests keep the tricky cases).
  - `compare_chat_template.py`: identical prompts to `apply_chat_template`.
  - `gen_jinja_cases.py`: the Jinja interpreter's expected outputs come from
    real Jinja with `transformers`' settings.
- **Perplexity**: `vkml-run --perplexity` scores a text token by token, and
  `tools/perplexity_hf.py` does the same with `transformers`. On a 250-token
  passage the two agree to four decimals. Quantized (with int8 activations,
  see Performance), `--q8` moves perplexity by under 0.5%:

  | Model | transformers (f32) | vkml | vkml `--q8` | vkml `--q4` |
  |---|---|---|---|---|
  | TinyLlama 1.1B | 6.8482 | 6.8482 | 6.8244 | 6.8275 |
  | SmolLM2 360M | 7.0729 | 7.0729 | 7.0882 | 8.3021 |

  On 1,636 tokens of the wikitext-2 test set, SmolLM2 360M scores 7.6412 with
  both, 7.6437 with `--q8` and 8.68 with `--q4`, which keeps the output
  projection at 8 bits: quantizing that to 4 bits too gave 9.83. Small models
  lose the most to 4 bits.

## Performance

On an Intel Iris Xe laptop GPU (integrated, shared LPDDR4x memory), plugged
in:

| Model | Load | Prompt (106 tokens) | Generation, bf16 | `--q8` | `--q4` |
|---|---|---|---|---|---|
| TinyLlama 1.1B | 2–5 s | ~120 tokens/s | ~22 tokens/s | ~34 tokens/s | ~49 tokens/s |
| SmolLM2 360M | 1–1.5 s | | ~47 tokens/s | ~65 tokens/s | ~76 tokens/s |
| Qwen2.5 0.5B | 2 s | | ~39 tokens/s | ~55 tokens/s | ~67 tokens/s |

Generation is bound by memory bandwidth: each token reads every weight once.
For TinyLlama that is 2.07 GB per token, which the matrix-vector kernels stream
at about 51 GB/s (40.6 ms of GPU time per token), the most this machine
sustains for large reads from uncached memory. The
matrix-vector kernel that does this reads each weight once for up to 8 rows of
input, so short prompts take the same path. Longer prompts use a tiled kernel
at about 390 GFLOP/s.

`--q8` stores each block of 32 weights as 32 bytes plus a 4-byte scale (1.1
GB per token for TinyLlama) and `--q4` as 16 bytes plus the scale (0.6 GB).
Unpacking them to floats as they are read, the matrix-vector kernels could not
issue loads fast enough to beat the 16-bit rate by much. On GPUs with
accelerated int8 dot products (`vkml-info` says), vkml instead quantizes the
activations to 8 bits too, once per layer input, and multiplies int8 by int8
four at a time, each lane loading 16 bytes of both operands. That took
TinyLlama from 28 to 49 tokens/s with `--q4`, and moves perplexity by under
0.5% (see Correctness). `ContextOptions::integer_dot_product = false` (or
`--no-dot`) turns it off.

Prompts with quantized weights take the same path through a tiled int8 kernel,
whose staged tiles hold four values per word and whose dot products do four
multiply-adds each. The f32 tiled kernel is held back by loading its tiles and
by its multiply-adds about equally, so this helps both:

| Prompt | bf16 | `--q8` | `--q4` |
|---|---|---|---|
| TinyLlama 1.1B, 512 tokens | ~165 tokens/s | ~345 tokens/s | ~460 tokens/s |
| Qwen2.5 0.5B, 222 tokens | ~415 tokens/s | ~1030 tokens/s | ~920 tokens/s |

The logits after those prompts stay within 2% (TinyLlama) and 5% (Qwen2.5) of
the largest one against `transformers` with `--q8`, the top ten tokens the
same; the int8 activations account for about 40% of that, and under 1% of perplexity.

To see where time goes, `vkml-run --profile` times every kernel with GPU
timestamps:

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
cannot flatter the bandwidth. Integrated GPUs change clocks a lot between runs,
so compare numbers over repeated runs.

## Limitations

- LLaMA-architecture decoders and Qwen2 only; no sliding-window attention
  (Mistral), MLP biases or mixture-of-experts yet, and configs asking for them
  are rejected.
- One sequence at a time; no batching of independent requests.
- Arithmetic is f32 (with f16/bf16, Q8_0 or Q4_0 weights); no k-quants or
  importance-weighted quantization, and no loading of pre-quantized (GGUF)
  files.
- Chat templates using Jinja beyond what chat templates commonly need (macros,
  tool-calling templates with complex logic) are rejected rather than
  approximated.

## License

MIT; see [LICENSE](LICENSE).
