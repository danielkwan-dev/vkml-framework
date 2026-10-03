#include "vkml/tokenizer.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "io/gguf.hpp"
#include "io/pretokenize.hpp"
#include "io/unicode.hpp"
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
    std::ostringstream text;
    text << in.rdbuf();
    init(text.str(), where);
}

namespace {

// BPE merges for a SentencePiece vocabulary, as transformers derives them
// when converting one to tokenizer.json: every split of a piece into two
// pieces, ranked by the piece's score, then by the longer left and right
// parts, then by piece, left and right ids.
nlohmann::json merges_from_scores(const std::vector<std::string>& pieces,
                                  const std::vector<float>& scores) {
    std::unordered_map<std::string_view, std::int32_t> ids;
    for (std::size_t id = 0; id < pieces.size(); ++id) ids.emplace(pieces[id], std::int32_t(id));
    struct Split {
        float score;
        std::size_t left_chars, right_chars;
        std::int32_t piece, left, right;
    };
    std::vector<Split> splits;
    for (std::size_t id = 0; id < pieces.size(); ++id) {
        const std::string_view piece = pieces[id];
        if (piece.empty()) continue;
        std::size_t chars = 0;
        for (std::size_t i = 0; i < piece.size(); i += utf8_length(std::uint8_t(piece[i]))) ++chars;
        std::size_t left_chars = 0;
        for (std::size_t at = utf8_length(std::uint8_t(piece[0])); at < piece.size();
             at += utf8_length(std::uint8_t(piece[at]))) {
            ++left_chars;
            const auto left = ids.find(piece.substr(0, at));
            const auto right = ids.find(piece.substr(at));
            if (left != ids.end() && right != ids.end()) {
                splits.push_back({scores[id], left_chars, chars - left_chars, std::int32_t(id),
                                  left->second, right->second});
            }
        }
    }
    std::ranges::sort(splits, [](const Split& a, const Split& b) {
        if (a.score != b.score) return a.score > b.score;
        if (a.left_chars != b.left_chars) return a.left_chars > b.left_chars;
        if (a.right_chars != b.right_chars) return a.right_chars > b.right_chars;
        return std::tie(a.piece, a.left, a.right) < std::tie(b.piece, b.left, b.right);
    });
    nlohmann::json merges = nlohmann::json::array();
    for (const Split& s : splits) {
        merges.push_back({pieces[std::size_t(s.left)], pieces[std::size_t(s.right)]});
    }
    return merges;
}

// The tokenizer.json a GGUF file's tokenizer came from, rebuilt from its
// metadata. llama.cpp's token types: 1 normal, 2 unknown, 3 control, 4 user
// defined, 5 unused, 6 byte.
nlohmann::json tokenizer_json_from_gguf(const nlohmann::json& m, const std::string& where) {
    using Json = nlohmann::json;
    const std::string model = m.value("tokenizer.ggml.model", std::string("?"));
    Json tokens = m.at("tokenizer.ggml.tokens");
    const Json types = m.value("tokenizer.ggml.token_type", Json::array());
    const auto type_of = [&](std::size_t id) {
        return id < types.size() ? types[id].get<int>() : 1;
    };
    Json merges = m.value("tokenizer.ggml.merges", Json{});
    if (merges.is_null() && model == "llama" && m.contains("tokenizer.ggml.scores")) {
        // A SentencePiece model as it was, which llama.cpp's converter
        // changes in two ways: user-defined pieces' ▁ become spaces, and they
        // and control pieces score -1000 rather than 0 (Gemma's). Then its
        // merges, as transformers makes them.
        std::vector<std::string> pieces;
        std::vector<float> scores = m.at("tokenizer.ggml.scores").get<std::vector<float>>();
        if (scores.size() != tokens.size()) {
            throw Error(where + ": has " + std::to_string(scores.size()) + " scores for " +
                        std::to_string(tokens.size()) + " tokens");
        }
        for (std::size_t id = 0; id < tokens.size(); ++id) {
            std::string piece = tokens[id].get<std::string>();
            if (type_of(id) == 4) {
                replace_all(piece, " ", "▁");
                tokens[id] = piece;
            }
            if (type_of(id) == 3 || type_of(id) == 4) scores[id] = 0.0f;
            pieces.push_back(std::move(piece));
        }
        merges = merges_from_scores(pieces, scores);
    }
    if (merges.is_null()) {
        throw Error(where +
                    ": has no tokenizer.ggml.merges, nor SentencePiece scores to make them "
                    "from");
    }

    Json vocab = Json::object();
    Json added = Json::array();
    for (std::size_t id = 0; id < tokens.size(); ++id) {
        const std::string token = tokens[id].get<std::string>();
        vocab[token] = id;
        const int type = type_of(id);
        if (type == 2 || type == 3 || type == 4) {
            added.push_back(
                {{"id", id}, {"content", token}, {"special", type != 4}, {"normalized", false}});
        }
    }
    const auto token_at = [&](const char* key) -> std::optional<std::string> {
        if (!m.contains(key)) return std::nullopt;
        const auto id = m.at(key).get<std::size_t>();
        if (id >= tokens.size()) return std::nullopt;
        return tokens[id].get<std::string>();
    };

    Json j;
    j["added_tokens"] = added;
    Json bpe{{"type", "BPE"}, {"vocab", vocab}, {"merges", merges}};
    bool add_bos;
    if (model == "llama") {
        // SentencePiece as HF's LLaMA tokenizer.json has it.
        const bool prefix = m.value("tokenizer.ggml.add_space_prefix", true);
        Json normalizers = Json::array();
        if (prefix) normalizers.push_back({{"type", "Prepend"}, {"prepend", "\u2581"}});
        normalizers.push_back(
            {{"type", "Replace"}, {"pattern", {{"String", " "}}}, {"content", "\u2581"}});
        j["normalizer"] = {{"type", "Sequence"}, {"normalizers", normalizers}};
        j["pre_tokenizer"] = nullptr;
        // Decoding strips the prefix space again, where there is one.
        if (prefix) {
            j["decoder"] =
                Json::object({{"type", "Strip"}, {"content", " "}, {"start", 1}, {"stop", 0}});
        }
        bpe["byte_fallback"] = true;
        bpe["fuse_unk"] = true;
        if (const auto unk = token_at("tokenizer.ggml.unknown_token_id")) bpe["unk_token"] = *unk;
        add_bos = m.value("tokenizer.ggml.add_bos_token", true);
    } else if (model == "gpt2") {
        const std::string pre = m.value("tokenizer.ggml.pre", std::string("default"));
        bool add_bos_default = false;  // unless the pre-tokenizer's model adds one
        Json byte_level = Json::object();
        byte_level["type"] = "ByteLevel";
        byte_level["add_prefix_space"] = false;
        byte_level["trim_offsets"] = true;
        byte_level["use_regex"] = true;
        const auto sequence = [](const Json& first, const Json& second) {
            Json seq = Json::object();
            seq["type"] = "Sequence";
            seq["pretokenizers"] = Json::array({first, second});
            return seq;
        };
        // A Split with the regex, then ByteLevel without its own.
        const auto split = [&](std::string_view pattern) {
            Json step = Json::object();
            step["type"] = "Split";
            step["pattern"] = Json::object({{"Regex", std::string(pattern)}});
            step["behavior"] = "Isolated";
            step["invert"] = false;
            Json no_regex = byte_level;
            no_regex["use_regex"] = false;
            return sequence(step, no_regex);
        };
        j["normalizer"] = nullptr;
        if (pre == "default" || pre == "gpt-2" || pre == "gpt2") {
            j["pre_tokenizer"] = byte_level;
        } else if (pre == "llama3" || pre == "llama-bpe" || pre == "smaug-bpe") {
            j["pre_tokenizer"] = split(detail::kLlama3Pattern);
            // As HF's LLaMA 3 files (and SmolLM3's, whose files llama.cpp calls
            // smaug-bpe). LLaMA 3 starts with a BOS token; SmolLM3 has none.
            bpe["ignore_merges"] = true;
            add_bos_default = pre != "smaug-bpe";
        } else if (pre == "dbrx") {
            // OLMo 2's (cl100k's merges): LLaMA 3's split, with no BOS and
            // without ignore_merges, as its HF tokenizer.
            j["pre_tokenizer"] = split(detail::kLlama3Pattern);
        } else if (pre == "qwen2") {
            j["pre_tokenizer"] = split(detail::kQwen2Pattern);
            j["normalizer"] = Json::object({{"type", "NFC"}});
        } else if (pre == "smollm") {
            Json digits = Json::object();
            digits["type"] = "Digits";
            digits["individual_digits"] = true;
            j["pre_tokenizer"] = sequence(digits, byte_level);
        } else {
            throw Error(
                where + ": the pre-tokenizer " + pre +
                " is not implemented (default, llama-bpe, smaug-bpe, dbrx, qwen2 and smollm are)");
        }
        add_bos = m.value("tokenizer.ggml.add_bos_token", add_bos_default);
    } else {
        throw Error(where + ": tokenizer model " + model +
                    " is not implemented (llama and gpt2 are)");
    }
    j["model"] = bpe;

    // BOS first, as a TemplateProcessing post-processor.
    const auto bos = token_at("tokenizer.ggml.bos_token_id");
    if (add_bos && bos) {
        Json special = Json::object();
        special["SpecialToken"] = Json::object({{"id", *bos}, {"type_id", 0}});
        Json sequence = Json::object();
        sequence["Sequence"] = Json::object({{"id", "A"}, {"type_id", 0}});
        Json entry = Json::object();
        entry["id"] = *bos;
        entry["ids"] = Json::array({m.at("tokenizer.ggml.bos_token_id")});
        entry["tokens"] = Json::array({*bos});
        Json post = Json::object();
        post["type"] = "TemplateProcessing";
        post["single"] = Json::array({special, sequence});
        post["special_tokens"] = Json::object({{*bos, entry}});
        j["post_processor"] = post;
    }
    return j;
}

}  // namespace

Tokenizer Tokenizer::from_gguf(const std::filesystem::path& path) {
    const std::string where = "tokenizer: " + path.string();
    const detail::Gguf file{path};
    Tokenizer t;
    try {
        t.init(tokenizer_json_from_gguf(file.metadata(), where).dump(), where);
    } catch (const nlohmann::json::exception& e) {
        throw Error(where + ": malformed tokenizer metadata: " + e.what());
    }
    return t;
}

void Tokenizer::init(std::string_view json_text, const std::string& where) {
    nlohmann::json json;
    try {
        json = nlohmann::json::parse(json_text);
    } catch (const nlohmann::json::exception& e) {
        throw Error(where + ": not valid JSON: " + e.what());
    }

    try {
        const auto& model = json.at("model");
        const std::string type = model.value("type", "");
        if (type != "BPE") throw Error(where + ": model type " + type + " is not implemented");
        for (const char* key : {"continuing_subword_prefix", "end_of_word_suffix"}) {
            // Qwen2 writes them as empty strings, which is the same as none.
            if (model.contains(key) && !model[key].is_null() && model[key] != "") {
                throw Error(where + ": BPE with " + key + " is not implemented");
            }
        }
        // Pre-tokenizer: none (SentencePiece-style), or byte-level, possibly
        // preceded by isolating digits or by a Split with a known regex.
        // Splits on a string, which only a normalizer that replaces every
        // occurrence makes harmless (checked below).
        std::vector<std::string> string_splits;
        std::size_t steps = 0;
        if (const auto& pre = json.value("pre_tokenizer", nlohmann::json{}); !pre.is_null()) {
            const auto add_step = [&](const nlohmann::json& p) {
                const std::string kind = p.at("type").get<std::string>();
                ++steps;
                if (kind == "Split" && p.at("pattern").contains("String")) {
                    string_splits.push_back(p["pattern"]["String"].get<std::string>());
                } else if (kind == "Digits" && p.value("individual_digits", false)) {
                    pre_steps_.push_back(PreStep::IsolateDigits);
                } else if (kind == "Split" && p.at("pattern").contains("Regex") &&
                           // Matches isolated, or kept with the rest removed
                           // (OLMo 2): the same for these regexes, which
                           // match every character.
                           (p.value("behavior", "") == "Isolated"  ? !p.value("invert", false)
                            : p.value("behavior", "") == "Removed" ? p.value("invert", false)
                                                                   : false)) {
                    const auto regex = p["pattern"]["Regex"].get<std::string>();
                    if (regex == detail::kGpt2Pattern) {
                        pre_steps_.push_back(PreStep::SplitGpt2);
                    } else if (regex == detail::kLlama3Pattern) {
                        pre_steps_.push_back(PreStep::SplitLlama3);
                    } else if (regex == detail::kQwen2Pattern) {
                        pre_steps_.push_back(PreStep::SplitQwen2);
                    } else if (regex == detail::kO200kPattern) {
                        pre_steps_.push_back(PreStep::SplitO200k);
                    } else {
                        throw Error(where + ": Split with regex " + regex + " is not implemented");
                    }
                } else if (kind == "Metaspace") {
                    // With split, each ▁-started word would be a BPE word of
                    // its own; SentencePiece files set it false.
                    if (p.value("split", true)) {
                        throw Error(where + ": Metaspace with split is not implemented");
                    }
                    // Older files write add_prefix_space instead of prepend_scheme.
                    const std::string scheme = p.value(
                        "prepend_scheme", p.value("add_prefix_space", true) ? "always" : "never");
                    using Prepend = Metaspace::Prepend;
                    const Prepend prepend = scheme == "first"   ? Prepend::First
                                            : scheme == "never" ? Prepend::Never
                                                                : Prepend::Always;
                    metaspace_ = Metaspace{prepend, p.value("replacement", "\u2581")};
                } else if (kind == "ByteLevel") {
                    byte_level_ = true;
                    if (p.value("add_prefix_space", false))
                        pre_steps_.push_back(PreStep::PrefixSpace);
                    if (p.value("use_regex", true)) pre_steps_.push_back(PreStep::SplitGpt2);
                } else {
                    throw Error(where + ": pre_tokenizer " + kind + " is not implemented");
                }
            };
            if (pre.at("type") == "Sequence") {
                for (const auto& step : pre.at("pretokenizers")) add_step(step);
            } else {
                add_step(pre);
            }
            // Only harmless splits (Gemma's) are the same as no pre-tokenizer.
            if (!byte_level_ && !metaspace_ && steps != string_splits.size()) {
                throw Error(where + ": pre-tokenizers without ByteLevel are not implemented");
            }
        }

        // Normalizers: Prepend, Replace and NFC, alone or in a Sequence.
        const auto add_normalizer = [&](const nlohmann::json& n) {
            const std::string kind = n.at("type").get<std::string>();
            if (kind == "Prepend") {
                normalizers_.push_back({Normalizer::Kind::Prepend, "", n.at("prepend")});
            } else if (kind == "Replace" && n.at("pattern").contains("String")) {
                normalizers_.push_back(
                    {Normalizer::Kind::Replace, n["pattern"]["String"], n.at("content")});
            } else if (kind == "NFC") {
                normalizers_.push_back({Normalizer::Kind::Nfc, "", ""});
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
        // A split on a string the normalizers replace everywhere, and do not
        // put back, finds nothing to split (Gemma: spaces, after they became ▁).
        for (const std::string& s : string_splits) {
            bool gone = false;
            for (const Normalizer& n : normalizers_) {
                if (n.kind == Normalizer::Kind::Replace && n.pattern == s &&
                    n.content.find(s) == std::string::npos) {
                    gone = true;
                } else if (n.content.find(s) != std::string::npos) {
                    gone = false;  // prepended or replaced back in
                }
            }
            if (s.empty() || !gone) {
                throw Error(where + ": pre_tokenizer Split on the string \"" + s +
                            "\" is not implemented");
            }
        }

        // The leading spaces decoding strips: a Strip step's, or the one a
        // Metaspace decoder removes where its pre-tokenizer would add it.
        if (const auto& d = json.value("decoder", nlohmann::json{}); !d.is_null() && !byte_level_) {
            const auto look = [&](const nlohmann::json& step) {
                const std::string kind = step.value("type", "");
                if (kind == "Strip" && step.value("content", "") == " ") {
                    strip_spaces_ = step.value("start", std::size_t{0});
                } else if (kind == "Metaspace") {
                    const bool prefix = step.value("add_prefix_space", true);
                    strip_spaces_ =
                        step.value("prepend_scheme", prefix ? "always" : "never") != "never" ? 1
                                                                                             : 0;
                }
            };
            if (d.value("type", "") == "Sequence") {
                for (const auto& step : d.at("decoders")) look(step);
            } else {
                look(d);
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
        ignore_merges_ = model.value("ignore_merges", false);
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
        // LLaMA 3's files put the template in a Sequence after ByteLevel.
        const auto read_template = [&](const nlohmann::json& post) {
            if (post.value("type", "") != "TemplateProcessing") return;
            const auto& single = post.at("single");
            if (!single.empty() && single[0].contains("SpecialToken")) {
                const auto name = single[0]["SpecialToken"].at("id").get<std::string>();
                const auto& ids = post.at("special_tokens").at(name).at("ids");
                if (!ids.empty()) bos_id_ = ids[0].get<std::int32_t>();
            }
        };
        if (const auto& post = json.value("post_processor", nlohmann::json{}); post.is_object()) {
            if (post.value("type", "") == "Sequence") {
                for (const auto& step : post.at("processors")) read_template(step);
            } else {
                read_template(post);
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
        } else if (n.kind == Normalizer::Kind::Nfc) {
            s = detail::nfc(s);
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
        encode_segment(prepare(text.substr(start, at - start), start == 0), out);
        out.push_back(match->second);
        at += match->first.size();
        start = at;
    }
    encode_segment(prepare(text.substr(start), start == 0), out);
    return out;
}

std::string Tokenizer::prepare(std::string_view segment, bool first) const {
    std::string s = normalize(segment);
    if (!metaspace_ || s.empty()) return s;
    const Metaspace& m = *metaspace_;
    replace_all(s, " ", m.replacement);
    const bool prepend = m.prepend == Metaspace::Prepend::Always ||
                         (m.prepend == Metaspace::Prepend::First && first);
    if (prepend && !s.starts_with(m.replacement)) s.insert(0, m.replacement);
    return s;
}

void Tokenizer::encode_segment(std::string_view text, std::vector<std::int32_t>& out) const {
    if (!byte_level_) {  // SentencePiece-style: the whole text is one word
        encode_word(text, out);
        return;
    }
    // Byte-level: each pre-tokenizer step splits every piece further, then each
    // piece's bytes are written as characters and merged as one word.
    std::vector<std::string> pieces{std::string(text)};
    for (const PreStep step : pre_steps_) {
        std::vector<std::string> next;
        for (const std::string& piece : pieces) {
            switch (step) {
                case PreStep::PrefixSpace:
                    next.push_back(piece.starts_with(' ') ? piece : " " + piece);
                    break;
                case PreStep::IsolateDigits:
                    for (const auto p : detail::split_digits(piece)) next.emplace_back(p);
                    break;
                case PreStep::SplitGpt2:
                case PreStep::SplitLlama3:
                case PreStep::SplitQwen2:
                case PreStep::SplitO200k: {
                    const auto rule = step == PreStep::SplitGpt2     ? detail::SplitRule::Gpt2
                                      : step == PreStep::SplitLlama3 ? detail::SplitRule::Llama3
                                      : step == PreStep::SplitQwen2  ? detail::SplitRule::Qwen2
                                                                     : detail::SplitRule::O200k;
                    for (const auto p : detail::split_words(piece, rule)) next.emplace_back(p);
                    break;
                }
            }
        }
        pieces = std::move(next);
    }
    for (const std::string& piece : pieces) encode_word(detail::bytes_to_unicode(piece), out);
}

void Tokenizer::encode_word(std::string_view text, std::vector<std::int32_t>& out) const {
    if (ignore_merges_) {
        if (const auto id = token_id(text)) {
            out.push_back(*id);
            return;
        }
    }
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
        if (byte_level_) {
            // Added tokens are stored as plain text; the rest as byte characters.
            const bool added =
                std::ranges::any_of(added_, [&](const auto& a) { return a.second == id; });
            text += added ? piece : detail::unicode_to_bytes(piece);
            continue;
        }
        const auto byte = std::ranges::find(byte_ids_, id);
        if (byte != byte_ids_.end()) {
            text += static_cast<char>(byte - byte_ids_.begin());
        } else {
            std::string p = piece;
            replace_all(p, "\u2581", " ");
            text += p;
        }
    }
    // SentencePiece-style decoding drops the space the normalizer prepended.
    for (std::size_t i = 0; i < strip_spaces_ && !text.empty() && text.front() == ' '; ++i)
        text.erase(0, 1);
    return text;
}

}  // namespace vkml
