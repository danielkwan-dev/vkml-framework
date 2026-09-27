# vkml

A GPU tensor library and LLM inference engine written from scratch in C++20 on
**Vulkan compute**, so it runs on GPUs from any vendor with a Vulkan 1.2
driver, not only CUDA hardware. It is developed on an Intel laptop GPU, and CI
runs it on Mesa's lavapipe, a CPU implementation of Vulkan.

It loads Hugging Face checkpoints of LLaMA-architecture models, tokenizes and
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
- **Models**: `vkml::Llama` loads `config.json` and sharded `.safetensors`,
  keeps weights in their checkpoint precision (bf16/f16), and decodes through a
  KV cache.
- **Tokenizers**: SentencePiece-style BPE (LLaMA 1/2, TinyLlama) and
  byte-level BPE with the GPT-2 and LLaMA 3 pre-tokenization rules (SmolLM,
  LLaMA 3, GPT-2 family), read from `tokenizer.json`. Files using anything else
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
`--seed`, `--max-reply`, `--context`. `VKML_DEVICE=<name>` or `--device
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
  released when that work completes, with a cap so a long load cannot hold
  unbounded staging memory.
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

- **Unit tests** (116, Catch2) run under the Vulkan validation layers, and each
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
    token for token (TinyLlama 1.1B, SmolLM2 360M).
  - `compare_tokenizer.py`: identical ids and decoded text to the `tokenizers`
    library on 24 texts (scripts, emoji, digits, whitespace, special tokens)
    for SentencePiece, byte-level and LLaMA 3-style tokenizers.
  - `compare_chat_template.py`: identical prompts to `apply_chat_template`.
  - `gen_jinja_cases.py`: the Jinja interpreter's expected outputs come from
    real Jinja with `transformers`' settings.

## Performance

On an Intel Iris Xe laptop GPU (integrated, shared LPDDR4x memory), bf16
weights:

| Model | Load | Prompt (106 tokens) | Generation |
|---|---|---|---|
| TinyLlama 1.1B | 5 s | ~85 tokens/s | ~15 tokens/s |
| SmolLM2 360M | 1.5 s | | ~29 tokens/s |

Decoding matmuls run at 54–58 GB/s, close to this machine's memory bandwidth;
prompt matmuls reach about 390 GFLOP/s. `vkml-bench` measures the matmul
shapes of a LLaMA forward pass and checks every result. Integrated GPUs change
clocks a lot between runs, so compare numbers over repeated runs.

## Limitations

- LLaMA-architecture decoders only; no attention biases (Qwen), sliding-window
  attention (Mistral) or mixture-of-experts yet.
- One sequence at a time; no batching of independent requests.
- Arithmetic is f32 (with f16/bf16 weights); no quantized weights yet.
- Chat templates using Jinja beyond what chat templates commonly need (macros,
  tool-calling templates with complex logic) are rejected rather than
  approximated.

## License

MIT; see [LICENSE](LICENSE).
