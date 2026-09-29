#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace vkml {

// Turns text into token ids and back, from an HF tokenizer.json.
//
// Implements the two BPE designs LLaMA-family models use:
//  - SentencePiece-style (LLaMA 1 and 2, TinyLlama): the normalizer
//    prepends U+2581 (▁) and replaces spaces with it, BPE runs over the
//    characters of the whole text, and characters outside the vocabulary fall
//    back to <0xNN> byte tokens.
//  - Byte-level (GPT-2, LLaMA 3, SmolLM): a regex splits text into words
//    (GPT-2's or LLaMA 3's pattern, optionally after isolating digits), each
//    word's bytes are written as printable characters, and BPE runs per word.
// Anything else in the file (other normalizers, pre-tokenizers or regexes) is
// rejected when loading rather than tokenized wrongly.
//
// Output matches the tokenizers library applied to the file as written.
// transformers' LlamaTokenizer (non-legacy mode) differs for SentencePiece
// files in one respect: it adds no ▁ to text that already starts with a space
// or follows a special token written in the text.
class Tokenizer {
public:
    explicit Tokenizer(const std::filesystem::path& path);

    // Special tokens (<s>, </s>, ...) written in the text are matched as
    // they are. With add_bos, the post-processor's leading BOS token comes first.
    std::vector<std::int32_t> encode(std::string_view text, bool add_bos = true) const;

    // Text for ids, skipping special tokens and ids outside the vocabulary.
    std::string decode(std::span<const std::int32_t> ids) const;

    std::optional<std::int32_t> token_id(std::string_view token) const;
    std::optional<std::int32_t> bos_id() const noexcept { return bos_id_; }
    std::size_t vocab_size() const noexcept { return pieces_.size(); }

private:
    struct Normalizer {
        enum class Kind { Prepend, Replace, Nfc } kind;
        std::string pattern;  // Replace only
        std::string content;
    };
    struct Merge {
        std::uint32_t rank;
        std::int32_t result;
    };
    // One step of a byte-level pre-tokenizer, applied to every piece so far.
    enum class PreStep { IsolateDigits, SplitGpt2, SplitLlama3, SplitQwen2, PrefixSpace };

    std::string normalize(std::string_view text) const;
    // A segment of text between added tokens, normalized and, for Metaspace,
    // with spaces replaced; first says it starts the input.
    std::string prepare(std::string_view segment, bool first) const;
    void encode_segment(std::string_view text, std::vector<std::int32_t>& out) const;
    void encode_word(std::string_view word, std::vector<std::int32_t>& out) const;

    std::vector<std::string> pieces_;  // by id
    std::vector<bool> special_;        // by id
    std::unordered_map<std::string, std::int32_t> ids_;
    std::unordered_map<std::uint64_t, Merge> merges_;          // (left id, right id) -> merge
    std::vector<std::pair<std::string, std::int32_t>> added_;  // longest first
    std::vector<Normalizer> normalizers_;
    bool byte_level_ = false;  // otherwise SentencePiece-style
    // The Metaspace pre-tokenizer of newer SentencePiece-style files: spaces
    // become replacement, which also starts each segment (Always), the first
    // one (First) or none (Never), unless the segment already starts with it.
    struct Metaspace {
        enum class Prepend { Always, First, Never } prepend;
        std::string replacement;
    };
    std::optional<Metaspace> metaspace_;
    std::vector<PreStep> pre_steps_;
    bool ignore_merges_ = false;                // a word already in the vocabulary is one token
    std::array<std::int32_t, 256> byte_ids_{};  // <0xNN> tokens, or -1
    std::optional<std::int32_t> unk_id_;
    std::optional<std::int32_t> bos_id_;
};

}  // namespace vkml
