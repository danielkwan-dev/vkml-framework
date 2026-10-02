# vkml

A GPU tensor library and LLM inference engine written from scratch in C++20 on
**Vulkan compute**, so it runs on GPUs from any vendor with a Vulkan 1.2
driver, not only CUDA hardware. It is developed on an Intel laptop GPU, and CI
runs it on Mesa's lavapipe, a CPU implementation of Vulkan.

It loads Hugging Face checkpoints of LLaMA-architecture models (and Mistral,
Qwen2, Qwen3, Gemma 2, Gemma 3, OLMo 2, Granite 3, SmolLM3 and Phi-3/Phi-4, which add sliding windows, attention
biases, per-head q and k norms, soft-capped scores and more), tokenizes and formats chat prompts the way
`transformers` does, and generates text on the GPU:

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
(Chat), SmolLM2 360M (base and Instruct), Qwen2.5 0.5B, Qwen3 0.6B, Llama 3.2
1B, Gemma 2 2B and Gemma 3 1B. Gemma 3 4B runs from a GGUF file, with only
its tokenizer and chat template checked, as `transformers` needs more memory
for it than the development machine has (see [Correctness](#correctness)).

## Features

- **Tensors on the GPU**: f32, f16, bf16 and i32; asynchronous execution with
  automatic lifetime management.
- **Operators**: elementwise arithmetic, SiLU/GeLU, softmax, RMSNorm, matmul
  (batched, transposed, 16-bit weights), permute/transpose, embedding lookup,
  rotary position embeddings (including LLaMA 3.1 scaling), and scaled
  dot-product attention with causal masking, grouped-query attention and a KV
  cache.
- **Models**: `vkml::Llama` runs LLaMA-architecture models (LLaMA 1-3.2,
  TinyLlama, SmolLM), Mistral, Qwen2/Qwen2.5, Qwen3, Gemma 2, Gemma 3's
  text models (1B, and the text model inside 4B and up), Granite 3 (from
  safetensors: Granite 3.3 2B Instruct with `--q8` scores 5.53 on that
  passage's 915 tokens), SmolLM3 (3B: 7.07 on its 704 tokens with `--q8`, 7.06
  as ggml-org's Q8_0 GGUF file), Phi-3 and Phi-4 (Phi-4-mini with `--q8`:
  7.10 on 708 tokens; fused projections are split as they load, and its
  tokenizer's o200k split, GPT-4o's, matches `transformers` on the 26
  texts) and OLMo 2 (from
  safetensors: OLMo 2 1B Instruct scores 9.1048 on the 3,000-character
  wikitext passage, where `transformers` in bf16 gives 9.1129). It loads `config.json` (as written by
  `transformers` 4 or 5) and sharded `.safetensors`, keeps weights in their
  checkpoint precision (bf16/f16) or quantizes them to 8 or 4 bits (Q8_0,
  Q4_0) on loading, and decodes through a KV cache in f32 or, for half the
  memory, f16. It also runs llama.cpp's GGUF files (see Usage).
- **Tokenizers**: SentencePiece-style BPE (LLaMA 1/2, TinyLlama, Gemma) and
  byte-level BPE with the GPT-2, LLaMA 3 and Qwen2 pre-tokenization rules
  (SmolLM, LLaMA 3, Qwen2, GPT-2 family), with NFC normalization, read from
  `tokenizer.json`. Files using anything else
  are rejected with an error naming it.
- **Chat templates**: a small Jinja interpreter runs the template each model
  ships in `tokenizer_config.json`, exactly as `apply_chat_template` does.
- **Sampling**: temperature, top-k, top-p, min-p and repetition penalty, in
  `transformers`' order, seeded; `generation_config.json` supplies a model's
  suggested settings and end-of-sequence tokens.
- **Command-line tools**: `vkml-chat` (interactive chat), `vkml-server`
  (OpenAI-compatible HTTP API, streaming), `vkml-run` (prompt completion), `vkml-info` (device report), `vkml-bench` (matmul benchmark),
  `vkml-tokenize`.

## Building

Requirements: CMake 3.25+, Ninja, a C++20 compiler, `glslc` (from shaderc or
the Vulkan SDK) and a Vulkan 1.2 driver. All C++ dependencies (Vulkan headers,
volk, vk-bootstrap, VMA, nlohmann/json, cpp-httplib, Catch2) are fetched and pinned at
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

A GGUF file, as llama.cpp uses and most quantized models are shared, works on
its own: its metadata holds the config, the tokenizer and the chat template.

```sh
vkml-chat --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

vkml reads GGUF versions 2 and 3 of LLaMA-architecture (including LLaMA 3.x
and Mistral), SmolLM3, Qwen2, Qwen3, Gemma 2 and Gemma 3 models. A SentencePiece tokenizer
stored with scores but no merges (Gemma's) gets the merges `transformers`
would make from them. Q8_0, Q4_0 and Q4_1 weights run as they are,
since they are vkml's own formats, with the file's f16 scales as they are.
q4_K weights run as Q4_1 too, since each 32-value sub-block has a scale and
an offset. Those are products of two of the file's numbers, which vkml
rounds to f16. Against exact f32 copies, Gemma 3 4B's perplexity moves by
under 0.4% (12.62 and 12.67 on the first 3,000 characters of the wikitext
passage), and the scales take half the memory. q2_K, q3_K, q5_0, q5_1, q5_K
and q6_K, which K-quant files also use for some layers, are decoded
on loading and run as Q8_0, finer than any of them: Llama 3.2 1B's Q3_K_L
file scores 10.09 on that passage, against 9.31 for its Q8_0 file. The
i-quants are rejected by name. The embedding table stays quantized
too, and with tied embeddings the output projection shares it; Gemma 3 4B's
262,144 x 2,560 table would not fit a GPU buffer of 1 GB as f16. Decoding
and repacking run on every core: Gemma 3 4B's Q4_K_M file loads in ~11 s and
generates at ~10.5 tokens/s. Llama 3.2 1B's Q4_K_M file loads in 3 s and
generates at ~33 tokens/s, against ~29 with its q4_K layers widened to Q8_0.

`vkml-chat` options: `--system`, `--temperature`, `--top-k`, `--top-p`,
`--min-p`, `--repetition-penalty`, `--seed`, `--max-reply`, `--context`, `--q8`
or `--q4`, `--kv-f16`, and `--no-think`, which asks reasoning models (Qwen3)
to reply without a `<think>` block first. Sampling not given on the command line comes from the
model's `generation_config.json` where it has any (Qwen2.5: temperature 0.7,
top-k 20, top-p 0.8, repetition penalty 1.1), else temperature 0.7 and top-p
0.9; the settings in use are printed at the start.
`VKML_DEVICE=<name>` or `--device <name>` picks a GPU by (part of) its name.

`vkml-server` serves a model over OpenAI's chat completions API, so the
`openai` client libraries, chat UIs and other tools that speak it can use vkml:

```sh
vkml-server --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf --port 8080
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
for chunk in client.chat.completions.create(
        model="qwen", messages=[{"role": "user", "content": "Hi!"}], stream=True):
    print(chunk.choices[0].delta.content or "", end="")
```

It serves `POST /v1/chat/completions` (streamed as server-sent events with
`"stream": true`), `POST /v1/completions` (a string `prompt` continued as
it is, without the chat template), `GET /v1/models` and `GET /health`, and reads `messages`,
`temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty`, `seed`,
`max_tokens`, `stop` and, as vLLM does, `chat_template_kwargs` with
`enable_thinking`; sampling a request leaves out defaults as in
`vkml-chat`. `tools`, assistant messages' `tool_calls` and `tool` messages go
to the chat template, and calls the model writes as Qwen2.5 and Qwen3 do
(`<tool_call>{"name": ..., "arguments": ...}</tool_call>`), or as Llama 3.x
does (a reply opening `{"name": ..., "parameters": ...}`), or as Mistral
does (a list of them, after `[TOOL_CALLS]`), come back as
`tool_calls`, with finish reason `tool_calls` (streamed whole, after any
text before them). Requests are answered one at a time, and the KV cache is kept
between them, so a conversation's next turn processes only its new messages.
Options: `--host` (127.0.0.1), `--port` (8080), `--context` (4096), `--q8` or
`--q4`, `--kv-f16`, `--device`, `--no-think` (for requests that do not set
`enable_thinking`), `--api-key <key>` to require that key
(as `Authorization: Bearer <key>`, which the `openai` clients send), and
`--chat-template <dir>`, which takes the template from a model directory's
`tokenizer_config.json` instead (a GGUF file may carry an older one: bartowski's
Mistral 7B v0.3 file has none for tools), and
`--reasoning-content`, which moves a reply's leading `<think>` block into
the message's `reasoning_content`, apart from its `content`, as DeepSeek's
API and vLLM's reasoning parsers do (streamed as `reasoning_content` deltas;
a template that opens the block itself is recognized).

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

- **Unit tests** (188, Catch2) run under the Vulkan validation layers, and each
  asserts the layers reported no errors. Operators are compared against
  double-precision references on the host; matmul uses the standard rounding
  bound for f32 dot products as its tolerance, so tests do not pass or fail by
  luck. Tests that guard subtle behaviour (deferred frees, rope scaling, rope
  layout) have been checked to fail when that behaviour is broken.
- **A tiny random LLaMA** is checked against a from-scratch double-precision
  implementation inside the test suite: prefill, token-by-token decoding
  through the KV cache, grouped-query attention, tied embeddings, bf16 weights,
  LLaMA 3.1 rope scaling, Qwen2's biases, Qwen3's q and k norms, sliding
  windows in every layer or some, and Gemma 2's and 3's layouts, Gemma 2's
  soft caps among them (set low enough to bend most scores and logits).
- **Against Hugging Face**, with the scripts in `tools/`:
  - `compare_hf.py`: last-token logits agree with `transformers` to within a
    few parts per million of the largest logit, and greedy generation matches
    token for token (TinyLlama 1.1B, SmolLM2 360M, Qwen2.5 0.5B, Qwen3 0.6B,
    Llama 3.2 1B, with its LLaMA 3.1 rope scaling, and `transformers`' own
    tiny random Mistral, converted to safetensors; with its sliding window
    set to 16 tokens, a 61-token prompt's logits still agree to 3e-7).
    Models too large for `transformers` in f32 here are checked on
    `slice_checkpoint.py`'s copy of their first few layers (a 4-layer slice
    of OLMo 2 1B agrees to 1.3e-6, 8 greedy tokens the same; one of
    Granite 3.3 2B to 2.5e-7, and one of SmolLM3 3B, rope skipped in its
    fourth layer, to 6.4e-7, each with the same perplexity to 6 digits; one
    of Phi-4-mini to 5.8e-7 once `transformers` has the embeddings vkml
    rounds to Q8_0, as too large for a GPU buffer, and `--no-dot` keeps
    activations in f32).
  - `compare_tokenizer.py`: identical ids and decoded text to the `tokenizers`
    library on 26 texts (scripts, emoji, digits, whitespace, special tokens,
    decomposed accents) for SentencePiece (old-style and Metaspace, as
    Mistral 7B v0.3), byte-level, LLaMA 3- and Qwen2-style tokenizers.
  - NFC normalization agrees with Python's `unicodedata` on every code point,
    alone and in its decomposed forms, and on 20,000 random sequences of marks
    (checked once while writing it; the unit tests keep the tricky cases).
  - `compare_chat_template.py`: identical prompts to `apply_chat_template`
    (TinyLlama, SmolLM2, Qwen2.5, Qwen3, Gemma 3 and Mistral 7B v0.3's
    templates, Qwen2.5's and Mistral's with their tool-calling branches, and
    the templates in TinyLlama's, Qwen2.5's, Qwen3's and Gemma 3's GGUF
    files). Qwen3's dropping of earlier `<think>` blocks is among the Jinja
    test cases.
  - GGUF files: TheBloke's TinyLlama Q8_0 and Q4_0 files, Qwen's Qwen2.5
    0.5B Q8_0 and Q4_K_M ones and unsloth's Llama 3.2 1B Q8_0 one load and
    run: their embedded tokenizers and chat templates match the HF ones on
    the 26 texts and 4 conversations; Q8_0 TinyLlama's logits are
    within 1.6% of the largest against `transformers` in f32, its greedy
    generation identical to `--q8` on the HF checkpoint; and on the passage
    above Qwen2.5 scores a perplexity of 9.65 (Q8_0) and 10.08 (Q4_K_M), where
    the HF checkpoint gives 9.45 and `--q4` 10.71; Llama 3.2's Q8_0 logits
    after it are within 1.1% of the largest against `transformers`, and it
    scores 8.94 (HF checkpoint), 8.94 (Q8_0), 9.31 (Q4_K_M) and 10.16 (`--q4`).
    ggml-org's Gemma 3 1B Q8_0 file, whose tokenizer has only scores, makes
    merges identical to the HF tokenizer's 514,906, in order, and encodes the
    26 texts identically. The k-quant and older
    block decoders match gguf-py, llama.cpp's Python package, on random
    blocks (`tools/gen_kquant_cases.py`).
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
  lose the most to 4 bits. On 1,467 tokens of it Qwen3 0.6B scores 11.1489
  with both, 11.1378 with `--q8`, 11.1101 as Qwen's Q8_0 GGUF file and 12.38
  with `--q4`. Gemma 3 1B, whose sliding window of 512 tokens those 1,428
  tokens (by its tokenizer) exceed, scores 14.9329 with both, 14.9504 with
  `--q8` and 14.9346 as ggml-org's Q8_0 GGUF file; its logits after a short
  prompt agree with `transformers` to 1.3e-6 of the largest, and 32 greedy
  tokens match. Gemma 3 4B, too large to run in `transformers` on this
  machine's 16 GB, was checked as ggml-org's Q4_K_M GGUF file: its tokenizer
  and chat template match the HF ones on the 26 texts and 4 conversations,
  and on the first 3,000 characters of that wikitext passage it scores 12.63
  where the 1B's Q8_0 file scores 20.48. Gemma 2 2B (unsloth's ungated
  copy), whose 256,000 x 2,304 bf16 embeddings exceed a 1 GB GPU buffer and
  are quantized to Q8_0 as they load, scores 8.7274 on 718 tokens of it, and
  bartowski's Q8_0 GGUF file 8.7357, where `transformers` in bf16 (f32 needs
  more memory than this machine has) gives 8.7370; the tokenizer and chat
  template, from either, match HF's on the 26 texts and 4 conversations
  (the template refuses a system message, and vkml refuses it too).
  Layers with a sliding window keep only its last rows and 1,024 more, as a
  ring buffer: Gemma 3 1B's KV cache at a context of 8,192 takes 136 MB
  instead of 436, and on 2,057 tokens of the wikitext passage, past that
  ring, it scores 18.3057, as `transformers` does.

## Performance

On an Intel Iris Xe laptop GPU (integrated, shared LPDDR4x memory), plugged
in:

| Model | Load | Prompt (106 tokens) | Generation, bf16 | `--q8` | `--q4` |
|---|---|---|---|---|---|
| TinyLlama 1.1B | 2–5 s | ~120 tokens/s | ~22 tokens/s | ~34 tokens/s | ~49 tokens/s |
| SmolLM2 360M | 1–1.5 s | | ~47 tokens/s | ~65 tokens/s | ~76 tokens/s |
| Qwen2.5 0.5B | 2 s | | ~39 tokens/s | ~55 tokens/s | ~67 tokens/s |
| Qwen3 0.6B | 1.5 s | | ~32 tokens/s | ~45 tokens/s | ~55 tokens/s |
| Llama 3.2 1B | 3 s | | ~18 tokens/s | ~29 tokens/s | ~34 tokens/s |
| Gemma 3 1B | 2–3 s | | ~18 tokens/s | ~26 tokens/s | ~35 tokens/s |
| Gemma 2 2B | 20 s | | ~7 tokens/s | ~13 tokens/s | ~13 tokens/s |

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

Attention for a generated token is one kernel split over chunks of 128 keys,
so a long cache gives the GPU many workgroups, then a small one merging the
chunks; as three matrix products with one query row each, it had far too
little parallel work. After a 1500-token prompt TinyLlama generates at ~20
tokens/s in bf16 and ~40 with `--q4` (it was ~25), against ~22 and ~49 with an
empty cache.

With `--kv-f16` the KV cache holds keys and values in f16: half the memory,
for about the same speed (the decoding attention kernel reads half the bytes
and widens them with the hardware's conversion, taking 4-5 ms per token at
1500 tokens either way); SmolLM2's wikitext perplexity moves from 7.6559 to
7.6527 with `--q8`. The default stays
f32, which keeps logits within parts per million of `transformers`.

A long prompt's attention scores would be large (288 MB per layer for 1500
tokens of TinyLlama), and allocating them every layer stalled the GPU for most
of the prompt. Queries now go in chunks whose scores stay under 64 MB, each
against only the keys it can see, which also skips most of the masked half:
a 1500-token prompt with `--q4` takes 5.3 s instead of 13.

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

`vkml-bench --decode <model>` times one decoding step's matrix-vector
products for a real model's shapes (only its config is read) in q8_0, q4_0
and q4_1. It runs four rounds, each timing every format, and keeps each
matrix's best time over all of them, because the GPU's clocks shift for tens
of seconds at a time. Across three runs for Gemma 3 4B, the per-token sums of
those bests agree to within 5%, while medians move by 10–20%. With
`--f32-scales` it times the f32 scales vkml used to keep: f16 ones save
5.5% of q8_0's bytes, 10% of q4_0's and 17% of q4_1's. On Gemma 3 4B's
shapes they take q8_0 from 86–92 ms per token to 82–84, q4_1 from 65 to
60, and q4_0 from 57 to 56.

## Limitations

- LLaMA-architecture decoders, Mistral, Qwen2, Qwen3, Gemma 2, Gemma 3's
  text models, OLMo 2, Granite 3, SmolLM3 and Phi-3/Phi-4 only, with SiLU or tanh GELU; no MLP biases or
  mixture-of-experts yet, and configs asking for them (or for exact GELU, or
  rope scaling other than LLaMA 3.1's and linear) are rejected. Gemma 3's
  image-text checkpoints run as text models only.
- Phi-3/Phi-4's LongRoPE takes its long factors for every position when
  `--context` is past the original 4,096 positions; `transformers` switches
  only once a sequence is that long, so shorter ones then differ slightly.
- One sequence at a time; no batching of independent requests.
- Arithmetic is f32 (with f16/bf16, Q8_0, Q4_0 or Q4_1 weights). GGUF
  layers in q5 and q6 formats run as Q8_0 (see Usage), and
  GGUF files that rescale rope other than linearly or as LLaMA 3.1 does
  (rope_freqs) are rejected.
- A weight matrix too large for one GPU buffer (often 1 GB on integrated
  GPUs, 4 GB on discrete ones), such as Gemma 2 2B's bf16 embeddings, is
  quantized to Q8_0 as it loads, whatever the options ask.
- Chat templates using Jinja beyond what chat templates commonly need (macros,
  tool-calling templates with complex logic) are rejected rather than
  approximated.

## License

MIT; see [LICENSE](LICENSE).
