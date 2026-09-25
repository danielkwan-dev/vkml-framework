// Tokenizes text with a tokenizer.json, for checking vkml against other
// implementations (see tools/compare_tokenizer.py).
//
//   vkml-tokenize <tokenizer.json> < texts.jsonl
//
// Each input line is a JSON string. Each output line is a JSON object with the
// ids (BOS included) and the text decoded back from them.

#include <cstdio>
#include <exception>
#include <iostream>
#include <string>

#include <nlohmann/json.hpp>
#include <vkml/tokenizer.hpp>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <tokenizer.json> < texts.jsonl\n", argv[0]);
        return 2;
    }
    try {
        const vkml::Tokenizer tokenizer{argv[1]};
        std::string line;
        while (std::getline(std::cin, line)) {
            const auto text = nlohmann::json::parse(line).get<std::string>();
            const auto ids = tokenizer.encode(text);
            const nlohmann::json out{{"ids", ids}, {"decoded", tokenizer.decode(ids)}};
            std::cout << out.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace) << '\n';
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
