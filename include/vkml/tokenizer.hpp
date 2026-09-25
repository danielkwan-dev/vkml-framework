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
// Implements SentencePiece-style BPE, the tokenizer of LLaMA 1 and 2,
// TinyLlama and Mistral: the normalizer prepends U+2581 (▁) and replaces
// spaces with it, BPE merges run over the characters of the whole text, and
// characters outside the vocabulary fall back to <0xNN> byte tokens. Other
// designs, such as the byte-level BPE of LLaMA 3 and GPT-2, are rejected when
// loading rather than tokenized wrongly.
//
// Output matches the tokenizers library applied to the file as written.
// transformers' LlamaTokenizer (non-legacy mode) differs in one respect: it
// adds no ▁ to text that already starts with a space or follows a special
// token written in the text.
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
        enum class Kind { Prepend, Replace } kind;
        std::string pattern;  // Replace only
        std::string content;
    };
    struct Merge {
        std::uint32_t rank;
        std::int32_t result;
    };

    std::string normalize(std::string_view text) const;
    void encode_segment(std::string_view text, std::vector<std::int32_t>& out) const;

    std::vector<std::string> pieces_;  // by id
    std::vector<bool> special_;        // by id
    std::unordered_map<std::string, std::int32_t> ids_;
    std::unordered_map<std::uint64_t, Merge> merges_;          // (left id, right id) -> merge
    std::vector<std::pair<std::string, std::int32_t>> added_;  // longest first
    std::vector<Normalizer> normalizers_;
    std::array<std::int32_t, 256> byte_ids_{};  // <0xNN> tokens, or -1
    std::optional<std::int32_t> unk_id_;
    std::optional<std::int32_t> bos_id_;
};

}  // namespace vkml
