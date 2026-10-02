#include "io/unicode.hpp"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <span>

namespace vkml::detail {

namespace {

struct CombiningClassRange {
    char32_t first, last;
    std::uint8_t combining_class;
};

// A code point's full canonical decomposition: kDecompositionData[offset..+length).
struct Decomposition {
    char32_t code_point;
    std::uint32_t offset, length;
};

// A pair that NFC composes into composite.
struct Composition {
    char32_t first, second, composite;
};

#include "io/unicode_tables.inc"

bool in_ranges(std::span<const CodepointRange> ranges, char32_t c) {
    // The ranges are sorted and disjoint: find the last one starting at or before c.
    const auto it =
        std::upper_bound(ranges.begin(), ranges.end(), c,
                         [](char32_t v, const CodepointRange& r) { return v < r.first; });
    return it != ranges.begin() && c <= std::prev(it)->last;
}

std::uint8_t combining_class(char32_t c) {
    const auto it =
        std::upper_bound(std::begin(kCombiningClasses), std::end(kCombiningClasses), c,
                         [](char32_t v, const CombiningClassRange& r) { return v < r.first; });
    if (it == std::begin(kCombiningClasses) || c > std::prev(it)->last) return 0;
    return std::prev(it)->combining_class;
}

// Hangul syllables are composed of jamo arithmetically (Unicode 3.12).
constexpr char32_t kSBase = 0xAC00, kLBase = 0x1100, kVBase = 0x1161, kTBase = 0x11A7;
constexpr char32_t kLCount = 19, kVCount = 21, kTCount = 28, kNCount = kVCount * kTCount;
constexpr char32_t kSCount = kLCount * kNCount;

void decompose(char32_t c, std::u32string& out) {
    if (c >= kSBase && c < kSBase + kSCount) {
        const char32_t s = c - kSBase;
        out += kLBase + s / kNCount;
        out += kVBase + s % kNCount / kTCount;
        if (s % kTCount != 0) out += kTBase + s % kTCount;
        return;
    }
    const auto it =
        std::lower_bound(std::begin(kDecompositions), std::end(kDecompositions), c,
                         [](const Decomposition& d, char32_t v) { return d.code_point < v; });
    if (it != std::end(kDecompositions) && it->code_point == c) {
        out.append(kDecompositionData + it->offset, it->length);
    } else {
        out += c;
    }
}

// The primary composite of first and second, or 0 if they do not compose.
char32_t compose(char32_t first, char32_t second) {
    if (first >= kLBase && first < kLBase + kLCount && second >= kVBase &&
        second < kVBase + kVCount) {
        return kSBase + ((first - kLBase) * kVCount + (second - kVBase)) * kTCount;
    }
    if (first >= kSBase && first < kSBase + kSCount && (first - kSBase) % kTCount == 0 &&
        second > kTBase && second < kTBase + kTCount) {
        return first + (second - kTBase);
    }
    const auto it = std::lower_bound(std::begin(kCompositions), std::end(kCompositions),
                                     std::pair{first, second},
                                     [](const Composition& c, std::pair<char32_t, char32_t> v) {
                                         return std::pair{c.first, c.second} < v;
                                     });
    const bool found = it != std::end(kCompositions) && it->first == first && it->second == second;
    return found ? it->composite : 0;
}

}  // namespace

std::string nfc(std::string_view text) {
    // Below U+0300 nothing decomposes or combines: most text takes this path.
    std::u32string chars;
    bool simple = true;
    for (std::size_t at = 0; at < text.size();) {
        const auto [c, length] = decode_utf8(text, at);
        chars += c;
        simple = simple && c < 0x300;
        at += length;
    }
    if (simple) return std::string(text);

    // Canonical decomposition, then canonical order: each run of marks
    // (nonzero combining class) stably sorted by class.
    std::u32string d;
    for (const char32_t c : chars) decompose(c, d);
    for (std::size_t i = 0; i < d.size();) {
        if (combining_class(d[i]) == 0) {
            ++i;
            continue;
        }
        std::size_t end = i;
        while (end < d.size() && combining_class(d[end]) != 0) ++end;
        std::stable_sort(
            d.begin() + std::ptrdiff_t(i), d.begin() + std::ptrdiff_t(end),
            [](char32_t a, char32_t b) { return combining_class(a) < combining_class(b); });
        i = end;
    }

    // Canonical composition (UAX #15): each character joins the last starter
    // unless a character between them blocks it, having the same or a higher
    // class, or being a starter itself.
    std::u32string out;
    std::size_t starter = std::u32string::npos;
    int last_class = -1;  // of the last character kept since the starter
    for (const char32_t c : d) {
        const int cc = combining_class(c);
        if (starter != std::u32string::npos && (last_class < cc || last_class == -1)) {
            if (const char32_t composite = compose(out[starter], c); composite != 0) {
                out[starter] = composite;
                continue;
            }
        }
        if (cc == 0) {
            starter = out.size();
            last_class = -1;
        } else {
            last_class = cc;
        }
        out += c;
    }

    std::string result;
    result.reserve(text.size());
    for (const char32_t c : out) append_utf8(result, c);
    return result;
}

bool is_letter(char32_t c) { return in_ranges(kLetterRanges, c); }

bool is_number(char32_t c) { return in_ranges(kNumberRanges, c); }

bool is_upperish(char32_t c) { return in_ranges(kUpperishRanges, c); }

bool is_lowerish(char32_t c) { return in_ranges(kLowerishRanges, c); }

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
