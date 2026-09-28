#include "io/pretokenize.hpp"

#include <algorithm>
#include <array>
#include <cstddef>

#include "io/unicode.hpp"

namespace vkml::detail {

const std::string_view kGpt2Pattern =
    R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)";
const std::string_view kLlama3Pattern =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
const std::string_view kQwen2Pattern =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

namespace {

struct CodePoint {
    char32_t c;
    std::size_t at;  // byte offset
};

std::vector<CodePoint> code_points(std::string_view text) {
    std::vector<CodePoint> out;
    for (std::size_t at = 0; at < text.size();) {
        const auto [c, length] = decode_utf8(text, at);
        out.push_back({c, at});
        at += length;
    }
    return out;
}

// Matches the patterns' alternatives in order, as the regex engine tries
// them, over code points.
class Scanner {
public:
    explicit Scanner(const std::vector<CodePoint>& cps) : cps_(cps), n_(cps.size()) {}

    // The end (exclusive) of the word the rule matches at i.
    std::size_t match(std::size_t i, SplitRule rule) const {
        const bool llama3 = rule != SplitRule::Gpt2;  // Qwen2 differs only in digits
        if (const std::size_t j = contraction(i, llama3)) return j;
        if (llama3) {
            // [^\r\n\p{L}\p{N}]?\p{L}+
            if (letter(i)) return run(i, &Scanner::letter);
            if (!newline(i) && !number(i) && letter(i + 1)) return run(i + 1, &Scanner::letter);
            // \p{N}{1,3}, or \p{N} for Qwen2
            if (number(i)) {
                return std::min(run(i, &Scanner::number), i + (rule == SplitRule::Qwen2 ? 1 : 3));
            }
            // ' ?[^\s\p{L}\p{N}]+[\r\n]*'
            if (const std::size_t j = optional_space_then(i, &Scanner::other)) {
                return run(j, &Scanner::newline);
            }
            // \s*[\r\n]+: whitespace up to and including its last line break
            if (space(i)) {
                const std::size_t end = run(i, &Scanner::space);
                for (std::size_t k = end; k > i; --k) {
                    if (newline(k - 1)) return k;
                }
            }
        } else {
            // ' ?\p{L}+', ' ?\p{N}+', ' ?[^\s\p{L}\p{N}]+'
            for (const Class cls : {&Scanner::letter, &Scanner::number, &Scanner::other}) {
                if (const std::size_t j = optional_space_then(i, cls)) return j;
            }
        }
        // \s+(?!\S) takes a whitespace run but leaves its last character to
        // start the following word; \s+ then takes a lone one.
        if (space(i)) {
            const std::size_t end = run(i, &Scanner::space);
            if (end == n_) return end;
            return end - i == 1 ? i + 1 : end - 1;
        }
        return i + 1;  // unreachable: every character is in one of the classes above
    }

private:
    using Class = bool (Scanner::*)(std::size_t) const;

    char32_t at(std::size_t i) const { return i < n_ ? cps_[i].c : 0; }
    bool letter(std::size_t i) const { return i < n_ && is_letter(at(i)); }
    bool number(std::size_t i) const { return i < n_ && is_number(at(i)); }
    bool space(std::size_t i) const { return i < n_ && is_whitespace(at(i)); }
    bool newline(std::size_t i) const { return at(i) == U'\r' || at(i) == U'\n'; }
    bool other(std::size_t i) const { return i < n_ && !space(i) && !letter(i) && !number(i); }

    std::size_t run(std::size_t i, Class cls) const {
        while ((this->*cls)(i)) ++i;
        return i;
    }

    // An optional space, then one or more of cls; 0 when there is no match.
    std::size_t optional_space_then(std::size_t i, Class cls) const {
        if ((this->*cls)(i)) return run(i, cls);
        if (at(i) == U' ' && (this->*cls)(i + 1)) return run(i + 1, cls);
        return 0;
    }

    // 's 't 're 've 'm 'll 'd, ignoring ASCII case for LLaMA 3; 0 when none.
    std::size_t contraction(std::size_t i, bool ignore_case) const {
        if (at(i) != U'\'') return 0;
        const auto lower = [&](std::size_t k) {
            const char32_t c = at(k);
            return ignore_case && c >= U'A' && c <= U'Z' ? c - U'A' + U'a' : c;
        };
        const char32_t a = lower(i + 1), b = lower(i + 2);
        if (a == U's' || a == U't' || a == U'm' || a == U'd') return i + 2;
        if ((a == U'r' && b == U'e') || (a == U'v' && b == U'e') || (a == U'l' && b == U'l')) {
            return i + 3;
        }
        return 0;
    }

    const std::vector<CodePoint>& cps_;
    std::size_t n_;
};

// GPT-2's bytes_to_unicode table and its inverse.
struct ByteMap {
    std::array<char32_t, 256> to_char{};
    std::array<int, 512> to_byte{};  // by code point; -1 where unused

    ByteMap() {
        to_byte.fill(-1);
        char32_t next = 256;
        for (int b = 0; b < 256; ++b) {
            const bool printable =
                (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE);
            to_char[std::size_t(b)] = printable ? char32_t(b) : next++;
            to_byte[to_char[std::size_t(b)]] = b;
        }
    }
};

const ByteMap& byte_map() {
    static const ByteMap map;
    return map;
}

}  // namespace

std::vector<std::string_view> split_words(std::string_view text, SplitRule rule) {
    const std::vector<CodePoint> cps = code_points(text);
    const Scanner scanner{cps};
    std::vector<std::string_view> words;
    for (std::size_t i = 0; i < cps.size();) {
        const std::size_t j = scanner.match(i, rule);
        const std::size_t end = j < cps.size() ? cps[j].at : text.size();
        words.push_back(text.substr(cps[i].at, end - cps[i].at));
        i = j;
    }
    return words;
}

std::vector<std::string_view> split_digits(std::string_view text) {
    std::vector<std::string_view> pieces;
    std::size_t start = 0;
    for (std::size_t at = 0; at < text.size();) {
        const auto [c, length] = decode_utf8(text, at);
        if (is_number(c)) {
            if (at > start) pieces.push_back(text.substr(start, at - start));
            pieces.push_back(text.substr(at, length));
            start = at + length;
        }
        at += length;
    }
    if (start < text.size()) pieces.push_back(text.substr(start));
    return pieces;
}

std::string bytes_to_unicode(std::string_view bytes) {
    std::string out;
    for (const char b : bytes) append_utf8(out, byte_map().to_char[static_cast<unsigned char>(b)]);
    return out;
}

std::string unicode_to_bytes(std::string_view text) {
    std::string out;
    for (std::size_t at = 0; at < text.size();) {
        const auto [c, length] = decode_utf8(text, at);
        const int b = c < 512 ? byte_map().to_byte[c] : -1;
        if (b >= 0) {
            out += static_cast<char>(b);
        } else {
            out.append(text.substr(at, length));
        }
        at += length;
    }
    return out;
}

}  // namespace vkml::detail
