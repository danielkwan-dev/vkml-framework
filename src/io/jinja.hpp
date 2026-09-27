#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

// A small Jinja interpreter for chat templates, the Jinja programs HF models
// ship in tokenizer_config.json to turn a conversation into prompt text.
//
// It covers what chat templates use: text, {{ }} output and {% %} statements
// (if/elif/else, for with loop variables and else, set, break, continue,
// generation blocks), with Jinja's whitespace control and the settings
// transformers renders with (trim_blocks, lstrip_blocks). Expressions have
// literals, variables, attributes, subscripts and slices, arithmetic, ~,
// comparisons, in, and/or/not, is-tests, conditional expressions, filters
// (trim, length, upper, lower, join, first, last, default, string, tojson)
// and string and dict methods, plus the globals raise_exception, range and
// strftime_now. Anything else is an error, not a silent approximation.
//
// Values are JSON values (keeping object keys in insertion order, as Python
// dicts do), so a conversation passes straight in.
namespace vkml::detail::jinja {

struct Node;

class Template {
public:
    explicit Template(std::string_view source);
    ~Template();
    Template(Template&&) noexcept;
    Template& operator=(Template&&) noexcept;

    // Renders with the given variables. Throws vkml::Error for template
    // errors, including a raise_exception call, whose message it carries.
    std::string render(const nlohmann::ordered_json& context) const;

private:
    std::vector<std::unique_ptr<Node>> nodes_;
};

}  // namespace vkml::detail::jinja
