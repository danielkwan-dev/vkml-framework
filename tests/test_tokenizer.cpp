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

#include "io/pretokenize.hpp"
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
    const auto whitespace =
        write_tokenizer("vkml_tok_whitespace.json", false, R"({"type": "Whitespace"})");
    REQUIRE_THROWS_WITH(Tokenizer{whitespace}, ContainsSubstring("Whitespace"));
}

namespace {

// A byte-level BPE tokenizer.json, like GPT-2's or SmolLM2's in miniature:
// ids 0..255 are the 256 byte characters (byte b is id b), then
//   256 Ġw  257 or  258 Ġwor  259 ld  260 Ġworld  261 He  262 ll  263 llo
//   264 Hello  265 <|endoftext|> (special)
std::filesystem::path write_byte_level_tokenizer(const std::string& file,
                                                 const std::string& pre_tokenizer,
                                                 const std::string& normalizer = "null",
                                                 const std::string& model_extra = "") {
    const auto quote = [](const std::string& s) {
        std::string out = "\"";
        for (const char c : s) {
            if (c == '"' || c == '\\') out += '\\';
            out += c;
        }
        return out + "\"";
    };
    std::string vocab;
    for (int b = 0; b < 256; ++b) {
        if (b) vocab += ", ";
        vocab += quote(vkml::detail::bytes_to_unicode(std::string(1, char(b)))) + ": " +
                 std::to_string(b);
    }
    const std::vector<std::string> pieces{"\u0120w", "or", "\u0120wor", "ld",   "\u0120world",
                                          "He",      "ll", "llo",       "Hello"};
    for (std::size_t i = 0; i < pieces.size(); ++i) {
        vocab += ", " + quote(pieces[i]) + ": " + std::to_string(256 + i);
    }
    const auto path = vkml_test::temp_path(file);
    std::ofstream(path, std::ios::binary) << R"({
  "added_tokens": [{"id": 265, "content": "<|endoftext|>", "special": true, "normalized": false}],
  "normalizer": )" << normalizer << R"(,
  "pre_tokenizer": )" << pre_tokenizer << R"(,
  "post_processor": null,
  "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true},
  "model": {"type": "BPE", "unk_token": null, "byte_fallback": false,)"
                                          << model_extra << R"(
    "vocab": {)" << vocab << R"(},
    "merges": ["\u0120 w", "o r", "\u0120w or", "l d", "\u0120wor ld", "H e", "l l", "ll o", "He llo"]}
})";
    return path;
}

const std::string kGpt2PreTokenizer = R"({"type": "Sequence", "pretokenizers": [
    {"type": "Digits", "individual_digits": true},
    {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true}]})";

}  // namespace

TEST_CASE("Byte-level tokenizer merges each pre-tokenized word", "[tokenizer]") {
    const Tokenizer tok{write_byte_level_tokenizer("vkml_bl.json", kGpt2PreTokenizer)};
    CHECK_FALSE(tok.bos_id().has_value());  // no post-processor: nothing is prepended
    // "Hello" and " world" are separate words; each merges to one token.
    CHECK(tok.encode("Hello world") == std::vector<std::int32_t>{264, 260});
    // Digits are split one by one before BPE; bytes map through GPT-2's table.
    CHECK(tok.encode("a12\n") == std::vector<std::int32_t>{'a', '1', '2', '\n'});
    CHECK(tok.encode("Hello<|endoftext|>") == std::vector<std::int32_t>{264, 265});
}

TEST_CASE("Byte-level tokenizer decodes bytes back, including split characters", "[tokenizer]") {
    const Tokenizer tok{write_byte_level_tokenizer("vkml_bl_decode.json", kGpt2PreTokenizer)};
    for (const std::string text : {"Hello world", "caf\u00e9 \U0001F642 42", "  two  spaces\n"}) {
        CAPTURE(text);
        CHECK(tok.decode(tok.encode(text)) == text);
    }
}

TEST_CASE("Byte-level tokenizer accepts the LLaMA 3 split and rejects unknown ones",
          "[tokenizer]") {
    const std::string pattern =
        R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
    const auto split = [](const std::string& regex) {
        std::string escaped;  // as a JSON string
        for (const char c : regex) {
            if (c == '"' || c == '\\') escaped += '\\';
            escaped += c;
        }
        return R"({"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": ")" +
               escaped + R"("}, "behavior": "Isolated", "invert": false},
            {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}]})";
    };
    const Tokenizer tok{write_byte_level_tokenizer("vkml_bl_llama3.json", split(pattern))};
    // \p{N}{1,3}: digits go in threes, then each to its byte token.
    CHECK(tok.encode("12345") == std::vector<std::int32_t>{'1', '2', '3', '4', '5'});
    CHECK(tok.encode("Hello world") == std::vector<std::int32_t>{264, 260});

    REQUIRE_THROWS_WITH(Tokenizer{write_byte_level_tokenizer("vkml_bl_bad.json", split("\\w+"))},
                        ContainsSubstring("Split"));
}

TEST_CASE("Byte-level tokenizer reads Qwen2's split, NFC normalizer and empty affixes",
          "[tokenizer]") {
    // As Qwen2 tokenizer.json files have them; the regex is JSON-escaped.
    const std::string pre = R"({"type": "Sequence", "pretokenizers": [
        {"type": "Split", "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},
         "behavior": "Isolated", "invert": false},
        {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}]})";
    const Tokenizer tok{write_byte_level_tokenizer(
        "vkml_bl_qwen2.json", pre, R"({"type": "NFC"})",
        R"( "continuing_subword_prefix": "", "end_of_word_suffix": "",)")};
    CHECK(tok.encode("Hello world") == std::vector<std::int32_t>{264, 260});
    // e + combining acute is normalized to U+00E9 (bytes C3 A9) first.
    CHECK(tok.encode("é") == std::vector<std::int32_t>{0xC3, 0xA9});
    CHECK(tok.encode("é") == std::vector<std::int32_t>{0xC3, 0xA9});
    // Special tokens are matched before normalizing, and kept.
    CHECK(tok.encode("<|endoftext|>é") == std::vector<std::int32_t>{265, 0xC3, 0xA9});
}
