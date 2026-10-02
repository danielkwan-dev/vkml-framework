#pragma once

#include <string>
#include <string_view>
#include <vector>

// Pre-tokenization for byte-level BPE tokenizers (GPT-2, LLaMA 3 and their
// descendants): splitting text into words before BPE, and GPT-2's mapping of
// bytes to printable characters.
namespace vkml::detail {

// The splitting regexes in use, written out as scanners because the C++
// standard library's regex has no Unicode classes:
//   Gpt2:   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//   Llama3: (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}
//           | ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//   Qwen2:  Llama3's with \p{N} for \p{N}{1,3}: every digit a word
enum class SplitRule { Gpt2, Llama3, Qwen2, O200k };

extern const std::string_view kGpt2Pattern;
extern const std::string_view kLlama3Pattern;
extern const std::string_view kO200kPattern;
extern const std::string_view kQwen2Pattern;

// The words text splits into; they cover it exactly, in order.
std::vector<std::string_view> split_words(std::string_view text, SplitRule rule);

// Splits text so every numeric character is a piece of its own, as the Digits
// pre-tokenizer does with individual_digits.
std::vector<std::string_view> split_digits(std::string_view text);

// GPT-2's reversible map from bytes to printable characters, which byte-level
// vocabularies are written in: printable Latin-1 maps to itself, other bytes
// to U+0100 and up (a space becomes Ġ).
std::string bytes_to_unicode(std::string_view bytes);

// The inverse; characters outside the map pass through as UTF-8.
std::string unicode_to_bytes(std::string_view text);

}  // namespace vkml::detail
