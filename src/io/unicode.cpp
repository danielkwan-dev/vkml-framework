#include "io/unicode.hpp"

#include <algorithm>
#include <span>

namespace vkml::detail {

namespace {

#include "io/unicode_tables.inc"

bool in_ranges(std::span<const CodepointRange> ranges, char32_t c) {
    // The ranges are sorted and disjoint: find the last one starting at or before c.
    const auto it =
        std::upper_bound(ranges.begin(), ranges.end(), c,
                         [](char32_t v, const CodepointRange& r) { return v < r.first; });
    return it != ranges.begin() && c <= std::prev(it)->last;
}

}  // namespace

bool is_letter(char32_t c) { return in_ranges(kLetterRanges, c); }

bool is_number(char32_t c) { return in_ranges(kNumberRanges, c); }

bool is_whitespace(char32_t c) {
    switch (c) {
        case 0x09:
        case 0x0A:
        case 0x0B:
        case 0x0C:
        case 0x0D:
        case 0x20:
        case 0x85:
        case 0xA0:
        case 0x1680:
        case 0x2028:
        case 0x2029:
        case 0x202F:
        case 0x205F:
        case 0x3000: return true;
        default: return c >= 0x2000 && c <= 0x200A;
    }
}

std::pair<char32_t, std::size_t> decode_utf8(std::string_view text, std::size_t at) {
    constexpr std::pair<char32_t, std::size_t> kInvalid{0xFFFD, 1};
    const auto byte = [&](std::size_t i) { return static_cast<unsigned char>(text[at + i]); };
    const unsigned char lead = byte(0);
    if (lead < 0x80) return {lead, 1};

    std::size_t length;
    char32_t c;
    if (lead >= 0xC2 && lead <= 0xDF) {
        length = 2;
        c = lead & 0x1F;
    } else if (lead >= 0xE0 && lead <= 0xEF) {
        length = 3;
        c = lead & 0x0F;
    } else if (lead >= 0xF0 && lead <= 0xF4) {
        length = 4;
        c = lead & 0x07;
    } else {
        return kInvalid;
    }
    if (at + length > text.size()) return kInvalid;
    for (std::size_t i = 1; i < length; ++i) {
        if ((byte(i) & 0xC0) != 0x80) return kInvalid;
        c = c << 6 | (byte(i) & 0x3F);
    }
    // Reject overlong forms, surrogates and values past U+10FFFF.
    const char32_t min = length == 2 ? 0x80 : length == 3 ? 0x800 : 0x10000;
    if (c < min || c > 0x10FFFF || (c >= 0xD800 && c <= 0xDFFF)) return kInvalid;
    return {c, length};
}

void append_utf8(std::string& out, char32_t c) {
    if (c < 0x80) {
        out += static_cast<char>(c);
    } else if (c < 0x800) {
        out += static_cast<char>(0xC0 | (c >> 6));
        out += static_cast<char>(0x80 | (c & 0x3F));
    } else if (c < 0x10000) {
        out += static_cast<char>(0xE0 | (c >> 12));
        out += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (c & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (c >> 18));
        out += static_cast<char>(0x80 | ((c >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (c & 0x3F));
    }
}

}  // namespace vkml::detail
