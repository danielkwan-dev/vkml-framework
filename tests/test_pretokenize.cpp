#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "io/pretokenize.hpp"

using vkml::detail::SplitRule;

namespace {

std::vector<std::string> split(std::string_view text, SplitRule rule) {
    std::vector<std::string> out;
    for (const std::string_view piece : vkml::detail::split_words(text, rule))
        out.emplace_back(piece);
    return out;
}

struct Case {
    std::string text;
    std::vector<std::string> gpt2;
    std::vector<std::string> llama3;
};

}  // namespace

// Expected pieces are what Python's regex module finds with each tokenizer's
// pattern (see tools/compare_tokenizer.py for the patterns in context).
TEST_CASE("split_words splits like the GPT-2 and LLaMA 3 pre-tokenizer regexes", "[pretokenize]") {
    const Case c = GENERATE(values<Case>({
        {"Hello world", {"Hello", " world"}, {"Hello", " world"}},
        {"I'm here, it's 12345 o'clock!",
         {"I", "'m", " here", ",", " it", "'s", " 12345", " o", "'", "clock", "!"},
         {"I", "'m", " here", ",", " it", "'s", " ", "123", "45", " o", "'clock", "!"}},
        {"a  b   \tc", {"a", " ", " b", "   ", "\t", "c"}, {"a", " ", " b", "   ", "\tc"}},
        {"  leading", {" ", " leading"}, {" ", " leading"}},
        {"trailing  ", {"trailing", "  "}, {"trailing", "  "}},
        {"line1\nline2\r\n\n  x",
         {"line", "1", "\n", "line", "2", "\r\n\n ", " x"},
         {"line", "1", "\n", "line", "2", "\r\n\n", " ", " x"}},
        {"HE'LL SHE'S", {"HE", "'", "LL", " SHE", "'", "S"}, {"HE", "'LL", " SHE", "'S"}},
        {"caf\u00e9 \u4e2d\u6587 \U0001F642ok",
         {"caf\u00e9", " \u4e2d\u6587", " \U0001F642", "ok"},
         {"caf\u00e9", " \u4e2d\u6587", " \U0001F642", "ok"}},
        {"x=1234567;", {"x", "=", "1234567", ";"}, {"x", "=", "123", "456", "7", ";"}},
        {"\t\n \n", {"\t\n \n"}, {"\t\n \n"}},
        {"", {}, {}},
    }));
    CAPTURE(c.text);
    CHECK(split(c.text, SplitRule::Gpt2) == c.gpt2);
    CHECK(split(c.text, SplitRule::Llama3) == c.llama3);
}

TEST_CASE("split_words splits like the Qwen2 regex: LLaMA 3's with single digits",
          "[pretokenize]") {
    CHECK(split("I'm here, it's 12345 o'clock!", SplitRule::Qwen2) ==
          std::vector<std::string>{"I", "'m", " here", ",", " it", "'s", " ", "1", "2", "3", "4",
                                   "5", " o", "'clock", "!"});
    CHECK(split("x=1234567;", SplitRule::Qwen2) ==
          std::vector<std::string>{"x", "=", "1", "2", "3", "4", "5", "6", "7", ";"});
    for (const std::string text : {"Hello world", "line1\nline2\r\n\n  x", "HE'LL SHE'S"}) {
        CAPTURE(text);
        CHECK(split(text, SplitRule::Qwen2) == split(text, SplitRule::Llama3));
    }
}

TEST_CASE("split_digits isolates every digit", "[pretokenize]") {
    std::vector<std::string> out;
    for (const auto piece : vkml::detail::split_digits("ab12 c\u0663d")) out.emplace_back(piece);
    CHECK(out == std::vector<std::string>{"ab", "1", "2", " c", "\u0663", "d"});
}

TEST_CASE("Byte-level mapping round-trips every byte through printable characters",
          "[pretokenize]") {
    // GPT-2's table: printable Latin-1 maps to itself, the rest to U+0100 on.
    CHECK(vkml::detail::bytes_to_unicode("Hi!") == "Hi!");
    CHECK(vkml::detail::bytes_to_unicode(" ") == "\u0120");   // the familiar Ġ
    CHECK(vkml::detail::bytes_to_unicode("\n") == "\u010a");  // Ċ
    std::string all;
    for (int b = 0; b < 256; ++b) all += static_cast<char>(b);
    CHECK(vkml::detail::unicode_to_bytes(vkml::detail::bytes_to_unicode(all)) == all);
}
