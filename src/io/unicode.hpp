#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <utility>

// The Unicode support tokenizers need: the character classes their
// pre-tokenization regexes use, NFC normalization and UTF-8 conversion.
namespace vkml::detail {

struct CodepointRange {
    char32_t first, last;
};

bool is_letter(char32_t c);      // general category L*, regex \p{L}
bool is_number(char32_t c);      // general category N*, regex \p{N}
bool is_whitespace(char32_t c);  // the White_Space property, regex \s
bool is_upperish(char32_t c);    // regex [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}] (o200k)
bool is_lowerish(char32_t c);    // regex [\p{Ll}\p{Lm}\p{Lo}\p{M}] (o200k)

// The code point starting at byte at, and its length in bytes. Malformed or
// truncated sequences decode as U+FFFD with length 1.
std::pair<char32_t, std::size_t> decode_utf8(std::string_view text, std::size_t at);

void append_utf8(std::string& out, char32_t c);

// text in Normalization Form C: canonically decomposed, marks in canonical
// order, then recomposed, as Qwen tokenizers normalize input. Invalid UTF-8
// becomes U+FFFD.
std::string nfc(std::string_view text);

}  // namespace vkml::detail
