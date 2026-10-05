# Usage

## Building

Requirements: CMake 3.25+, Ninja, a C++20 compiler, `glslc` (from shaderc or
the Vulkan SDK) and a Vulkan 1.2 driver. All C++ dependencies (Vulkan headers,
volk, vk-bootstrap, VMA, nlohmann/json, cpp-httplib, Catch2) are fetched and
pinned at configure time.

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

Presets:

| Preset | What it is |
|---|---|
| `debug` | validation layers on |
| `release` | optimized |
| `ci` | debug with warnings as errors |

Binaries land in `build/<preset>/apps/`. MinGW builds link their runtime
statically, so they also run outside the MSYS2 shell.

`cmake --install build/release --prefix <dir>` copies the tools to
`<dir>/bin`. The shaders are compiled into the binaries, so they need nothing
else.

### Tests

```sh
cmake --preset debug && cmake --build --preset debug && ctest --preset debug
```

The tests need a Vulkan device. Without a GPU, Mesa's lavapipe works (CI uses
it). On Windows the loader finds MSYS2's validation layers only with
`VK_ADD_LAYER_PATH=C:\msys64\ucrt64\bin` set; without them the validation
tests skip.

## Models

vkml reads two formats:

- **Hugging Face directories**: `config.json`, `*.safetensors` (sharded is
  fine), `tokenizer.json`, `tokenizer_config.json`, and optionally
  `generation_config.json`. For example:
  `huggingface_hub.snapshot_download("HuggingFaceTB/SmolLM2-360M-Instruct",
  local_dir="models/SmolLM2-360M-Instruct")`. `config.json` may be written
  by `transformers` 4 or 5.
- **GGUF files** (versions 2 and 3), as llama.cpp uses. The file's metadata
  holds the config, the tokenizer and the chat template, so it works on its
  own.

Supported architectures: LLaMA (1 to 3.2, TinyLlama, SmolLM), Mistral,
Qwen2/Qwen2.5, Qwen3, Gemma 2, Gemma 3 (text models), OLMo 2, Granite 3,
SmolLM3, Phi-3 and Phi-4.

### Weight formats

From a Hugging Face checkpoint, weights stay in their checkpoint precision
(bf16 or f16), or are quantized as they load:

- `--q8`: Q8_0 (blocks of 32 int8 weights with one scale).
- `--q4`: Q4_0 (blocks of 32 4-bit weights with one scale). The output
  projection stays at 8 bits.

From a GGUF file:

| GGUF type | How vkml runs it |
|---|---|
| Q8_0, Q4_0, Q4_1 | as they are (vkml's own formats, f16 scales kept) |
| q4_K | as Q4_1: each 32-value sub-block has a scale and an offset (products of two of the file's numbers, rounded to f16) |
| q2_K, q3_K, q5_0, q5_1, q5_K, q6_K | decoded on loading, run as Q8_0 (finer than any of them) |
| i-quants | rejected by name |

The embedding table stays quantized. With tied embeddings, the output
projection shares it. (Gemma 3 4B's 262,144 x 2,560 table would not fit a
1 GB GPU buffer as f16.) A SentencePiece tokenizer stored with scores but no
merges (Gemma's) gets the merges `transformers` would make from them.

GGUF files may carry their own chat template, sometimes older than the Hugging
Face one. vkml renders it faithfully; `--chat-template <dir>` takes the
template from a model directory instead.

## vkml-chat

```sh
vkml-chat --model models/SmolLM2-360M-Instruct            # /reset starts over, /quit leaves
vkml-chat --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

| Option | Meaning |
|---|---|
| `--system <text>` | system prompt |
| `--temperature`, `--top-k`, `--top-p`, `--min-p`, `--repetition-penalty` | sampling |
| `--seed <n>` | sampling seed |
| `--max-reply <n>` | tokens per reply |
| `--context <n>` | context length |
| `--q8`, `--q4` | quantize weights as they load |
| `--kv-f16` | KV cache in f16 (half the memory) |
| `--no-think` | ask reasoning models (Qwen3) to reply without a `<think>` block |
| `--device <name>` | pick a GPU by (part of) its name; also `VKML_DEVICE=<name>` |

Sampling not given on the command line comes from the model's
`generation_config.json` (Qwen2.5: temperature 0.7, top-k 20, top-p 0.8,
repetition penalty 1.1), else temperature 0.7 and top-p 0.9. The settings in
use are printed at the start.

## vkml-run

```sh
vkml-run --model models/TinyLlama-1.1B-Chat-v1.0 --prompt "The capital of France is" --generate 64
vkml-run --model <model> --tokens 1,450,7483 --generate 16
vkml-run --model <model> --prompt "$(cat passage.txt)" --perplexity
```

| Option | Meaning |
|---|---|
| `--prompt <text>` / `--tokens <id,id,...>` | the input, as text or token ids |
| `--generate <n>` | tokens to generate (greedy) |
| `--top <n>` | print the top n next tokens |
| `--ignore-eos` | keep generating past end-of-sequence |
| `--perplexity` | score the input instead of continuing it |
| `--profile` | GPU time per kernel (see [performance.md](performance.md)) |
| `--dump-logits <file>` | write the logits, for `tools/compare_hf.py` |
| `--no-dot` | no int8 activations (see [performance.md](performance.md)) |
| `--context`, `--q8`, `--q4`, `--kv-f16`, `--device` | as in `vkml-chat` |

## vkml-server

An OpenAI-compatible HTTP server, so the `openai` client libraries, chat UIs
and evaluation harnesses can use vkml.

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

### Endpoints

| Endpoint | |
|---|---|
| `POST /v1/chat/completions` | chat, streamed as server-sent events with `"stream": true` |
| `POST /v1/completions` | a `prompt` (text or token ids) continued as it is, without the chat template |
| `GET /v1/models` | the loaded model |
| `GET /health` | liveness |

### Request fields

- Messages and sampling: `messages`, `temperature`, `top_p`, `top_k`, `min_p`,
  `repetition_penalty`, `frequency_penalty` and `presence_penalty` (these two
  count only the reply's tokens), `seed`, `max_tokens`, `stop`.
- `stream_options.include_usage`: a last chunk with the token counts.
- `chat_template_kwargs.enable_thinking`, as vLLM does.
- `logprobs` (and `top_logprobs` for chat, up to 20): each reply token's log
  probability, in each API's shape. They come from the model's own
  distribution before temperature and other sampling settings, as vLLM does by
  default.
- `echo` (completions): repeats the prompt with its tokens' log probabilities
  (the first is `null`), from one pass over it. `max_tokens` may then be 0.
  This is what lm-evaluation-harness's `local-completions` sends to score
  answers.
- Sampling a request leaves out falls back as in `vkml-chat`.

Settings out of range, and a `response_format` other than text, get a 400
rather than being ignored. Bytes of a reply that are not whole UTF-8
characters (one cut off by `max_tokens`, say) come back as U+FFFD, as
`transformers` decodes them.

### Tool calls

`tools`, assistant messages' `tool_calls` and `tool` messages go to the chat
template. Calls the model writes in any of these formats come back as
`tool_calls`, with finish reason `tool_calls`:

- Qwen2.5 and Qwen3: `<tool_call>{"name": ..., "arguments": ...}</tool_call>`
- Llama 3.x: a reply opening `{"name": ..., "parameters": ...}`
- Mistral: a list of calls after `[TOOL_CALLS]`

When streamed, a call is sent whole, after any text before it.

### Options

| Option | Default | Meaning |
|---|---|---|
| `--host` | 127.0.0.1 | |
| `--port` | 8080 | |
| `--context` | 4096 | |
| `--q8`, `--q4`, `--kv-f16`, `--device` | | as in `vkml-chat` |
| `--no-think` | | for requests that do not set `enable_thinking` |
| `--api-key <key>` | | require `Authorization: Bearer <key>` (the `openai` clients send it) |
| `--chat-template <dir>` | | take the template from a model directory's `tokenizer_config.json` (bartowski's Mistral 7B v0.3 GGUF file has none for tools) |
| `--reasoning-content` | | move a reply's leading `<think>` block into `reasoning_content`, apart from `content`, as DeepSeek's API and vLLM's reasoning parsers do; streamed as `reasoning_content` deltas; a template that opens the block itself is recognized |

Requests are answered one at a time. The KV cache is kept between them, so a
conversation's next turn processes only its new messages.

### Evaluation with lm-evaluation-harness

```sh
vkml-server --model models/Qwen2.5-0.5B-Instruct --port 8080
lm_eval --model local-completions --tasks arc_easy --model_args \
  model=qwen,base_url=http://127.0.0.1:8080/v1/completions,tokenizer=models/Qwen2.5-0.5B-Instruct
```

## Other tools

- `vkml-info`: the Vulkan devices, the one vkml picks, and its features (such
  as accelerated int8 dot products).
- `vkml-bench`: times a LLaMA forward pass's matmul shapes (see
  [performance.md](performance.md)).
- `vkml-tokenize <tokenizer.json | model.gguf> < texts.jsonl`: token ids and
  decoded text for each line, for `tools/compare_tokenizer.py`.

## As a library

From a CMake project, which builds vkml (and its shaders) along with it. The
tests, tools and examples are left out when vkml is not the top-level project.

```cmake
include(FetchContent)
FetchContent_Declare(vkml GIT_REPOSITORY https://github.com/danielkwan-dev/vkml-framework
                         GIT_TAG main)  # or add_subdirectory(path/to/vkml-framework)
FetchContent_MakeAvailable(vkml)
target_link_libraries(my_app PRIVATE vkml::vkml)
```

[examples/quickstart.cpp](../examples/quickstart.cpp) runs a full chat turn:
tensors, loading a model, the chat template, and the sampling loop.
