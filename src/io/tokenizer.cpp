#include "vkml/tokenizer.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <limits>

#include <nlohmann/json.hpp>

#include "vkml/error.hpp"

namespace vkml {

namespace {

std::uint64_t pair_key(std::int32_t left, std::int32_t right) {
    return std::uint64_t(std::uint32_t(left)) << 32 | std::uint32_t(right);
}

// Bytes in the UTF-8 sequence that starts with lead; invalid leads count as 1.
std::size_t utf8_length(unsigned char lead) {
    if (lead >= 0xF0 && lead < 0xF8) return 4;
    if (lead >= 0xE0) return lead < 0xF0 ? 3 : 1;
    if (lead >= 0xC0) return 2;
    return 1;
}

void replace_all(std::string& s, const std::string& from, const std::string& to) {
    if (from.empty()) return;
    for (std::size_t at = s.find(from); at != std::string::npos;
         at = s.find(from, at + to.size())) {
        s.replace(at, from.size(), to);
    }
}

}  // namespace

Tokenizer::Tokenizer(const std::filesystem::path& path) {
    const std::string where = "tokenizer: " + path.string();
    std::ifstream in(path, std::ios::binary);
    if (!in) throw Error(where + ": cannot open file");
    nlohmann::json json;
    try {
        json = nlohmann::json::parse(in);
    } catch (const nlohmann::json::exception& e) {
        throw Error(where + ": not valid JSON: " + e.what());
    }

    try {
        const auto& model = json.at("model");
        const std::string type = model.value("type", "");
        if (type != "BPE") throw Error(where + ": model type " + type + " is not implemented");
        for (const char* key : {"continuing_subword_prefix", "end_of_word_suffix"}) {
            if (model.contains(key) && !model[key].is_null()) {
                throw Error(where + ": BPE with " + key + " is not implemented");
            }
        }
        if (json.contains("pre_tokenizer") && !json["pre_tokenizer"].is_null()) {
            throw Error(where + ": pre_tokenizer " +
                        json["pre_tokenizer"].value("type", std::string("?")) +
                        " is not implemented; only SentencePiece-style BPE is");
        }

        // Normalizers: Prepend and Replace, alone or in a Sequence.
        const auto add_normalizer = [&](const nlohmann::json& n) {
            const std::string kind = n.at("type").get<std::string>();
            if (kind == "Prepend") {
                normalizers_.push_back({Normalizer::Kind::Prepend, "", n.at("prepend")});
            } else if (kind == "Replace" && n.at("pattern").contains("String")) {
                normalizers_.push_back(
                    {Normalizer::Kind::Replace, n["pattern"]["String"], n.at("content")});
            } else {
                throw Error(where + ": normalizer " + kind + " is not implemented");
            }
        };
        if (const auto& n = json.value("normalizer", nlohmann::json{}); !n.is_null()) {
            if (n.at("type") == "Sequence") {
                for (const auto& step : n.at("normalizers")) add_normalizer(step);
            } else {
                add_normalizer(n);
            }
        }

        // Vocabulary and added tokens share one id space.
        const auto set_piece = [&](const std::string& piece, std::int32_t id) {
            if (id < 0) throw Error(where + ": negative token id for " + piece);
            if (std::size_t(id) >= pieces_.size()) {
                pieces_.resize(std::size_t(id) + 1);
                special_.resize(std::size_t(id) + 1);
            }
            pieces_[std::size_t(id)] = piece;
            ids_[piece] = id;
        };
        for (const auto& [piece, id] : model.at("vocab").items())
            set_piece(piece, id.get<std::int32_t>());
        for (const auto& added : json.value("added_tokens", nlohmann::json::array())) {
            const auto id = added.at("id").get<std::int32_t>();
            const auto content = added.at("content").get<std::string>();
            set_piece(content, id);
            special_[std::size_t(id)] = added.value("special", false);
            added_.emplace_back(content, id);
        }
        std::ranges::sort(
            added_, [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });

        byte_ids_.fill(-1);
        if (model.value("byte_fallback", false)) {
            char name[8];
            for (int b = 0; b < 256; ++b) {
                std::snprintf(name, sizeof name, "<0x%02X>", b);
                if (const auto id = token_id(name)) byte_ids_[std::size_t(b)] = *id;
            }
        }
        if (model.contains("unk_token") && model["unk_token"].is_string()) {
            unk_id_ = token_id(model["unk_token"].get<std::string>());
        }

        // Merges: "left right" strings or [left, right] pairs, in rank order.
        std::uint32_t rank = 0;
        for (const auto& m : model.at("merges")) {
            std::string left, right;
            if (m.is_array()) {
                left = m.at(0).get<std::string>();
                right = m.at(1).get<std::string>();
            } else {
                const auto text = m.get<std::string>();
                const auto space = text.find(' ');
                if (space == std::string::npos)
                    throw Error(where + ": malformed merge \"" + text + "\"");
                left = text.substr(0, space);
                right = text.substr(space + 1);
            }
            const auto l = token_id(left), r = token_id(right), result = token_id(left + right);
            if (l && r && result) merges_.try_emplace(pair_key(*l, *r), Merge{rank, *result});
            ++rank;
        }

        // The BOS token is the post-processor template's leading special token.
        if (const auto& post = json.value("post_processor", nlohmann::json{});
            !post.is_null() && post.value("type", "") == "TemplateProcessing") {
            const auto& single = post.at("single");
            if (!single.empty() && single[0].contains("SpecialToken")) {
                const auto name = single[0]["SpecialToken"].at("id").get<std::string>();
                const auto& ids = post.at("special_tokens").at(name).at("ids");
                if (!ids.empty()) bos_id_ = ids[0].get<std::int32_t>();
            }
        }
    } catch (const nlohmann::json::exception& e) {
        throw Error(where + ": malformed: " + e.what());
    }
}

std::optional<std::int32_t> Tokenizer::token_id(std::string_view token) const {
    const auto it = ids_.find(std::string(token));
    return it == ids_.end() ? std::nullopt : std::optional{it->second};
}

std::string Tokenizer::normalize(std::string_view text) const {
    std::string s(text);
    for (const Normalizer& n : normalizers_) {
        if (n.kind == Normalizer::Kind::Prepend) {
            if (!s.empty()) s.insert(0, n.content);  // as HF: empty text stays empty
        } else {
            replace_all(s, n.pattern, n.content);
        }
    }
    return s;
}

std::vector<std::int32_t> Tokenizer::encode(std::string_view text, bool add_bos) const {
    std::vector<std::int32_t> out;
    if (add_bos && bos_id_) out.push_back(*bos_id_);

    // Split around added tokens (longest match first); the text between them
    // is normalized and tokenized segment by segment, as HF does.
    std::size_t start = 0;
    for (std::size_t at = 0; at < text.size();) {
        const auto match = std::ranges::find_if(added_, [&](const auto& a) {
            return !a.first.empty() && text.substr(at, a.first.size()) == a.first;
        });
        if (match == added_.end()) {
            ++at;
            continue;
        }
        encode_segment(normalize(text.substr(start, at - start)), out);
        out.push_back(match->second);
        at += match->first.size();
        start = at;
    }
    encode_segment(normalize(text.substr(start)), out);
    return out;
}

void Tokenizer::encode_segment(std::string_view text, std::vector<std::int32_t>& out) const {
    // One symbol per UTF-8 character: its token id, or -1 with its bytes kept
    // for byte fallback when the vocabulary lacks it.
    struct Symbol {
        std::int32_t id;
        std::string_view bytes;
    };
    std::vector<Symbol> symbols;
    for (std::size_t at = 0; at < text.size();) {
        const std::size_t len =
            std::min(utf8_length(static_cast<unsigned char>(text[at])), text.size() - at);
        const std::string_view ch = text.substr(at, len);
        symbols.push_back({token_id(ch).value_or(-1), ch});
        at += len;
    }

    // Merge the lowest-ranked adjacent pair, leftmost first, until none is left.
    for (;;) {
        std::size_t best = symbols.size();
        std::uint32_t best_rank = std::numeric_limits<std::uint32_t>::max();
        std::int32_t best_result = -1;
        for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
            if (symbols[i].id < 0 || symbols[i + 1].id < 0) continue;
            const auto it = merges_.find(pair_key(symbols[i].id, symbols[i + 1].id));
            if (it != merges_.end() && it->second.rank < best_rank) {
                best = i;
                best_rank = it->second.rank;
                best_result = it->second.result;
            }
        }
        if (best == symbols.size()) break;
        symbols[best] = {best_result, {}};
        symbols.erase(symbols.begin() + std::ptrdiff_t(best) + 1);
    }

    for (const Symbol& s : symbols) {
        if (s.id >= 0) {
            out.push_back(s.id);
            continue;
        }
        const bool fallback = std::ranges::all_of(
            s.bytes, [&](char c) { return byte_ids_[static_cast<unsigned char>(c)] >= 0; });
        if (fallback) {
            for (const char c : s.bytes) out.push_back(byte_ids_[static_cast<unsigned char>(c)]);
        } else if (unk_id_) {
            out.push_back(*unk_id_);
        }
    }
}

std::string Tokenizer::decode(std::span<const std::int32_t> ids) const {
    std::string text;
    for (const std::int32_t id : ids) {
        if (id < 0 || std::size_t(id) >= pieces_.size() || special_[std::size_t(id)]) continue;
        const std::string& piece = pieces_[std::size_t(id)];
        const auto byte = std::ranges::find(byte_ids_, id);
        if (byte != byte_ids_.end()) {
            text += static_cast<char>(byte - byte_ids_.begin());
        } else {
            std::string p = piece;
            replace_all(p, "▁", " ");
            text += p;
        }
    }
    if (!text.empty() && text.front() == ' ') text.erase(0, 1);  // the prepended ▁
    return text;
}

}  // namespace vkml
