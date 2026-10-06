#include <fstream>
#include <string>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <nlohmann/json.hpp>

#include <vkml/error.hpp>

#include "io/jinja.hpp"

using Catch::Matchers::ContainsSubstring;
using vkml::detail::jinja::Template;

// Each case in tests/data/jinja_cases.json holds a template, its variables and
// the output (or error) real Jinja gives with transformers' settings; see
// tools/gen_jinja_cases.py.
TEST_CASE("Templates render as Jinja renders them", "[jinja]") {
    std::ifstream in(VKML_TEST_DATA_DIR "/jinja_cases.json", std::ios::binary);
    REQUIRE(in);
    const auto cases = nlohmann::ordered_json::parse(in);
    REQUIRE(cases.size() > 30);

    for (const auto& c : cases) {
        const std::string name = c.at("name");
        CAPTURE(name);
        const Template t{c.at("template").get<std::string>()};
        if (c.contains("error")) {
            REQUIRE_THROWS_WITH(t.render(c.at("context")),
                                ContainsSubstring(c.at("error").get<std::string>()));
        } else {
            CHECK(t.render(c.at("context")) == c.at("output").get<std::string>());
        }
    }
}

TEST_CASE("Templates reject what they do not support, naming it", "[jinja]") {
    REQUIRE_THROWS_WITH(Template{"{% macro m() %}x{% endmacro %}"}, ContainsSubstring("macro"));
    REQUIRE_THROWS_WITH(Template{"{{ x | wordwrap }}"}.render({{"x", "a"}}),
                        ContainsSubstring("wordwrap"));
    REQUIRE_THROWS_WITH(Template{"{% if true %}unclosed"}, ContainsSubstring("endif"));
    REQUIRE_THROWS_WITH(Template{"{{ 'unterminated }}"}, ContainsSubstring("unclosed"));
    REQUIRE_THROWS_WITH(Template{"{{ 1 + }}"}, ContainsSubstring("expected"));
}

TEST_CASE("Templates bound their nesting and range sizes", "[jinja]") {
    // A model file's template is untrusted: nesting deep enough to overflow the
    // stack, or a range too large for memory, must fail as an error.
    const auto repeat = [](const std::string& s, int n) {
        std::string out;
        for (int i = 0; i < n; ++i) out += s;
        return out;
    };
    const int deep = 1000;
    CHECK_THROWS_WITH(Template{"{{ " + repeat("(", deep) + "1" + repeat(")", deep) + " }}"},
                      ContainsSubstring("deep"));
    CHECK_THROWS_WITH(Template{"{{ " + repeat("not ", deep) + "x }}"}, ContainsSubstring("deep"));
    CHECK_THROWS_WITH(Template{"{{ " + repeat("-", deep) + "1 }}"}, ContainsSubstring("deep"));
    CHECK_THROWS_WITH(Template{"{{ " + repeat("+", deep) + "1 }}"}, ContainsSubstring("deep"));
    CHECK_THROWS_WITH(Template{repeat("{% if true %}", deep) + repeat("{% endif %}", deep)},
                      ContainsSubstring("deep"));
    // Nesting real templates use still works.
    CHECK(Template{"{{ " + repeat("(", 50) + "1" + repeat(")", 50) + " }}"}.render({}) == "1");

    CHECK_THROWS_WITH(Template{"{{ range(2000000) | length }}"}.render({}),
                      ContainsSubstring("range"));
    CHECK_THROWS_WITH(
        Template{"{{ range(-9223372036854775807, 9223372036854775807) | length }}"}.render({}),
        ContainsSubstring("range"));
    CHECK(Template{"{{ range(1000) | length }}"}.render({}) == "1000");
    CHECK(Template{"{{ range(9223372036854775800, 9223372036854775807, 3) | list }}"}.render({}) ==
          "[9223372036854775800, 9223372036854775803, 9223372036854775806]");
    CHECK(Template{"{{ range(5, 0, -2) | list }}"}.render({}) == "[5, 3, 1]");
}
