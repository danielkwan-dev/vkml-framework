#include <string>
#include <utility>

#include <catch2/catch_test_macros.hpp>

#include "io/unicode.hpp"

using namespace vkml::detail;

TEST_CASE("Unicode classes match the regex classes tokenizers use", "[unicode]") {
    for (const char32_t c : {U'A', U'z', U'\u00e9', U'\u03a9', U'\u4e2d', U'\u3042', U'\u05d0'}) {
        CAPTURE(std::uint32_t(c));
        CHECK(is_letter(c));
        CHECK_FALSE(is_number(c));
    }
    // Decimal digits in several scripts, a vulgar fraction and a Roman numeral.
    for (const char32_t c : {U'0', U'9', U'\u0663', U'\u00bd', U'\u216b'}) {
        CAPTURE(std::uint32_t(c));
        CHECK(is_number(c));
        CHECK_FALSE(is_letter(c));
    }
    for (const char32_t c : {U' ', U'\t', U'\n', U'\r', U'\u00a0', U'\u3000', U'\u2028'}) {
        CAPTURE(std::uint32_t(c));
        CHECK(is_whitespace(c));
    }
    for (const char32_t c : {U'!', U'-', U'_', U'\U0001F642', U'\u20ac'}) {
        CAPTURE(std::uint32_t(c));
        CHECK_FALSE(is_letter(c));
        CHECK_FALSE(is_number(c));
        CHECK_FALSE(is_whitespace(c));
    }
}

TEST_CASE("UTF-8 decoding reads each code point and survives invalid bytes", "[unicode]") {
    const std::string text = "a\u00e9\u4e2d\U0001F642";
    CHECK(decode_utf8(text, 0) == std::pair<char32_t, std::size_t>{U'a', 1});
    CHECK(decode_utf8(text, 1) == std::pair<char32_t, std::size_t>{U'\u00e9', 2});
    CHECK(decode_utf8(text, 3) == std::pair<char32_t, std::size_t>{U'\u4e2d', 3});
    CHECK(decode_utf8(text, 6) == std::pair<char32_t, std::size_t>{U'\U0001F642', 4});

    // A lone continuation byte, and a lead byte cut off by the end.
    CHECK(decode_utf8("\x80", 0) == std::pair<char32_t, std::size_t>{U'\uFFFD', 1});
    CHECK(decode_utf8("\xE4\xB8", 0) == std::pair<char32_t, std::size_t>{U'\uFFFD', 1});

    std::string out;
    for (const char32_t c : {U'a', U'\u00e9', U'\u4e2d', U'\U0001F642'}) append_utf8(out, c);
    CHECK(out == text);
}
