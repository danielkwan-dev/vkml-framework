# Changelog

## Unreleased

Security fixes from a review of the server and the model-file parsers:

- vkml-server no longer sends `Access-Control-Allow-Origin: *`, which let any
  web page use a server on the user's machine; `--cors-origin` allows one
  origin. API keys are compared in constant time, and serving beyond
  localhost without `--api-key` prints a warning.
- GGUF and safetensors files whose tensor sizes, offsets or alignment wrap
  past 64 bits are rejected; such a file could pass the bounds check.
- Chat templates, which come with model files, may nest at most 128 levels
  (1,000 nested parentheses overflowed the stack) and `range()` makes at
  most 2^20 values.
- CI's GitHub Actions are pinned to commit hashes.

## 0.2.0 (2026-10-05)

First release.

- **Models**: LLaMA 1-3.2 (TinyLlama, SmolLM2), Mistral, Qwen2/Qwen2.5,
  Qwen3, Gemma 2, Gemma 3 (text), OLMo 2, Granite 3, SmolLM3, Phi-3 and Phi-4,
  from Hugging Face checkpoints or GGUF files (versions 2 and 3).
- **Weights**: bf16/f16 as stored, Q8_0 and Q4_0 quantization on loading,
  and GGUF's Q8_0, Q4_0, Q4_1 and K-quant formats. int8 dot-product kernels on
  GPUs that accelerate them.
- **Runtime**: asynchronous command stream with pooled buffers, matrix-vector
  kernels for decoding, tiled kernels for prefill, decoding attention split
  over chunks of keys, a sliding-window ring-buffer KV cache, and an f16 KV
  cache option.
- **Text**: SentencePiece and byte-level BPE tokenizers with NFC
  normalization, a Jinja interpreter for chat templates, and sampling
  (temperature, top-k, top-p, min-p, repetition penalty).
- **Tools**: `vkml-chat`, `vkml-run` (generation, perplexity, profiling),
  `vkml-server` (OpenAI-compatible: chat and completions, streaming, tool
  calls, logprobs, reasoning content), `vkml-info`, `vkml-bench`,
  `vkml-tokenize`.
- **Checks**: 220 tests under the Vulkan validation layers, CI on Mesa's
  lavapipe and a Windows build, and scripts comparing logits, perplexity,
  tokenizers and chat templates with Hugging Face `transformers`.
