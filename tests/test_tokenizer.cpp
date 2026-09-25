#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <vkml/tokenizer.hpp>

#include "support/safetensors_writer.hpp"

using Catch::Matchers::ContainsSubstring;
using vkml::Tokenizer;

namespace {

// A SentencePiece-style BPE tokenizer.json, like LLaMA 2's in miniature:
//   0 <unk>, 1 <s>, 2 </s>, 3..258 the byte tokens <0x00>..<0xFF>, then
//   259 ▁  260 h  261 e  262 l  263 o  264 w  265 r  266 d  267 ▁h  268 ll
//   269 ▁he  270 llo  271 ▁hello  272 ▁w  273 or  274 ▁wor  275 ld
const std::vector<std::string> kPieces{"▁",  "h",   "e",   "l",      "o",  "w",  "r",    "d", "▁h",
                                       "ll", "▁he", "llo", "▁hello", "▁w", "or", "▁wor", "ld"};
const std::vector<std::pair<std::string, std::string>> kMerges{
    {"▁", "h"}, {"l", "l"}, {"▁h", "e"},  {"ll", "o"}, {"▁he", "llo"},
    {"▁", "w"}, {"o", "r"}, {"▁w", "or"}, {"l", "d"}};

std::filesystem::path write_tokenizer(const std::string& file, bool array_merges,
                                      const std::string& pre_tokenizer = "null") {
    std::string vocab = R"("<unk>": 0, "<s>": 1, "</s>": 2)";
    char byte_token[8];
    for (int b = 0; b < 256; ++b) {
        std::snprintf(byte_token, sizeof byte_token, "<0x%02X>", b);
        vocab += ", \"" + std::string(byte_token) + "\": " + std::to_string(3 + b);
    }
    for (std::size_t i = 0; i < kPieces.size(); ++i) {
        vocab += ", \"" + kPieces[i] + "\": " + std::to_string(259 + i);
    }
    std::string merges;
    for (const auto& [left, right] : kMerges) {
        if (!merges.empty()) merges += ", ";
        merges += array_merges ? "[\"" + left + "\", \"" + right + "\"]"
                               : "\"" + left + " " + right + "\"";
    }

    const auto path = vkml_test::temp_path(file);
    std::ofstream(path, std::ios::binary) << R"({
  "version": "1.0",
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true, "normalized": false},
    {"id": 1, "content": "<s>", "special": true, "normalized": false},
    {"id": 2, "content": "</s>", "special": true, "normalized": false}
  ],
  "normalizer": {"type": "Sequence", "normalizers": [
    {"type": "Prepend", "prepend": "▁"},
    {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}]},
  "pre_tokenizer": )" << pre_tokenizer << R"(,
  "post_processor": {"type": "TemplateProcessing",
    "single": [{"SpecialToken": {"id": "<s>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}],
    "special_tokens": {"<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]}}},
  "decoder": {"type": "Sequence", "decoders": [
    {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
    {"type": "ByteFallback"}, {"type": "Fuse"},
    {"type": "Strip", "content": " ", "start": 1, "stop": 0}]},
  "model": {"type": "BPE", "unk_token": "<unk>", "byte_fallback": true, "fuse_unk": true,
    "vocab": {)" << vocab << R"(},
    "merges": [)" << merges << R"(]}
})";
    return path;
}

}  // namespace

TEST_CASE("Tokenizer applies merges in rank order and adds the BOS token", "[tokenizer]") {
    const bool array_merges = GENERATE(false, true);  // both merge formats HF writes
    CAPTURE(array_merges);
    const Tokenizer tok{write_tokenizer("vkml_tok.json", array_merges)};

    CHECK(tok.vocab_size() == 276);
    CHECK(tok.bos_id() == 1);
    CHECK(tok.token_id("</s>") == 2);
    CHECK(tok.encode("hello world") == std::vector<std::int32_t>{1, 271, 274, 275});
    CHECK(tok.encode("hello world", false) == std::vector<std::int32_t>{271, 274, 275});
    CHECK(tok.encode("") == std::vector<std::int32_t>{1});
}

TEST_CASE("Tokenizer merges the leftmost of equally ranked pairs first", "[tokenizer]") {
    const Tokenizer tok{write_tokenizer("vkml_tok_tie.json", false)};
    // ▁ l l l: (l, l) occurs twice; merging the left pair leaves ▁ ll l.
    CHECK(tok.encode("lll", false) == std::vector<std::int32_t>{259, 268, 262});
}

TEST_CASE("Tokenizer falls back to byte tokens for text outside the vocabulary", "[tokenizer]") {
    const Tokenizer tok{write_tokenizer("vkml_tok_bytes.json", false)};
    // é is the two bytes C3 A9, tokens 3 + 0xC3 and 3 + 0xA9.
    CHECK(tok.encode("hé", false) == std::vector<std::int32_t>{267, 3 + 0xC3, 3 + 0xA9});
    CHECK(tok.decode(std::vector<std::int32_t>{267, 3 + 0xC3, 3 + 0xA9}) == "hé");
}

TEST_CASE("Tokenizer matches special tokens written in the text", "[tokenizer]") {
    const Tokenizer tok{write_tokenizer("vkml_tok_special.json", false)};
    // Each segment between specials is normalized on its own, so hi gets its ▁.
    CHECK(tok.encode("<s>hi</s>") == std::vector<std::int32_t>{1, 1, 267, 3 + 'i', 2});
}

TEST_CASE("Tokenizer decodes to the text it encoded", "[tokenizer]") {
    const Tokenizer tok{write_tokenizer("vkml_tok_decode.json", false)};
    for (const std::string text : {"hello world", "world hello", "  hello", "lll", "héllo w"}) {
        CAPTURE(text);
        CHECK(tok.decode(tok.encode(text)) == text);  // special tokens are skipped
    }
    CHECK(tok.decode(std::vector<std::int32_t>{271, 999}) == "hello");  // unknown ids are skipped
}

TEST_CASE("Tokenizer rejects tokenizers it does not implement", "[tokenizer]") {
    REQUIRE_THROWS_WITH(Tokenizer{vkml_test::temp_path("vkml_no_such_tokenizer.json")},
                        ContainsSubstring("vkml_no_such_tokenizer.json"));
    const auto byte_level =
        write_tokenizer("vkml_tok_bytelevel.json", false, R"({"type": "ByteLevel"})");
    REQUIRE_THROWS_WITH(Tokenizer{byte_level}, ContainsSubstring("ByteLevel"));
}
