#pragma once

#include <algorithm>
#include <cstddef>
#include <string>

// The length of the longest prefix of text that does not end inside a UTF-8
// sequence. Byte-level and byte-fallback tokens can split a character across
// decoding steps, so streamed text is printed only up to here.
inline std::size_t complete_utf8_prefix(const std::string& text) {
    for (std::size_t back = 1; back <= std::min<std::size_t>(3, text.size()); ++back) {
        const auto c = static_cast<unsigned char>(text[text.size() - back]);
        if ((c & 0xC0) == 0x80) continue;  // continuation byte: keep looking for the lead
        const std::size_t needed = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
        return needed > back ? text.size() - back : text.size();
    }
    return text.size();
}
