#include "io/jinja.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <ctime>
#include <map>
#include <optional>

#include "vkml/error.hpp"

namespace vkml::detail::jinja {

using Json = nlohmann::ordered_json;

namespace {

[[noreturn]] void fail(const std::string& message) { throw Error("chat template: " + message); }

// ------------------------------------------------------------------ values

// A JSON value, or Jinja's undefined (a missing variable, key or attribute),
// which renders as nothing and is falsy.
struct Value {
    Json v;
    bool undefined = false;
    bool is_namespace = false;  // from namespace(): {% set ns.attr = ... %} may change it
};

Value undefined() { return {Json(), true}; }

bool truthy(const Value& x) {
    if (x.undefined || x.v.is_null()) return false;
    if (x.v.is_boolean()) return x.v.get<bool>();
    if (x.v.is_number_integer()) return x.v.get<std::int64_t>() != 0;
    if (x.v.is_number()) return x.v.get<double>() != 0.0;
    if (x.v.is_string()) return !x.v.get_ref<const std::string&>().empty();
    return !x.v.empty();  // array or object
}

std::string format_float(double d) {
    char buf[64];
    const auto end = std::to_chars(buf, buf + sizeof buf, d).ptr;
    std::string s(buf, end);
    if (s.find_first_of(".eni") == std::string::npos) s += ".0";  // Python writes 2.0, not 2
    return s;
}

std::string text_of(const Json& v, bool quote_strings);

// Python's repr of a string: single quotes unless it contains one and no double.
std::string python_repr(const std::string& s) {
    const char q =
        s.find('\'') != std::string::npos && s.find('"') == std::string::npos ? '"' : '\'';
    std::string out(1, q);
    for (const char c : s) {
        if (c == q || c == '\\') out += '\\';
        if (c == '\n') {
            out += "\\n";
        } else {
            out += c;
        }
    }
    return out + q;
}

// How Jinja prints a value: Python's str(), with repr() inside containers.
std::string text_of(const Json& v, bool quote_strings) {
    if (v.is_null()) return "None";
    if (v.is_boolean()) return v.get<bool>() ? "True" : "False";
    if (v.is_number_integer()) return std::to_string(v.get<std::int64_t>());
    if (v.is_number()) return format_float(v.get<double>());
    if (v.is_string())
        return quote_strings ? python_repr(v.get<std::string>()) : v.get<std::string>();
    std::string out = v.is_array() ? "[" : "{";
    bool first = true;
    if (v.is_array()) {
        for (const auto& e : v) {
            out += (first ? "" : ", ") + text_of(e, true);
            first = false;
        }
        return out + "]";
    }
    for (const auto& [k, e] : v.items()) {
        out += (first ? "" : ", ") + python_repr(k) + ": " + text_of(e, true);
        first = false;
    }
    return out + "}";
}

std::string text_of(const Value& x) { return x.undefined ? "" : text_of(x.v, false); }

// json.dumps with its default separators ", " and ": ", keeping non-ASCII.
std::string to_json(const Json& v) {
    if (v.is_array()) {
        std::string out = "[";
        for (std::size_t i = 0; i < v.size(); ++i) out += (i ? ", " : "") + to_json(v[i]);
        return out + "]";
    }
    if (v.is_object()) {
        std::string out = "{";
        bool first = true;
        for (const auto& [k, e] : v.items()) {
            out += (first ? "" : ", ") + Json(k).dump() + ": " + to_json(e);
            first = false;
        }
        return out + "}";
    }
    return v.dump();
}

std::int64_t as_int(const Value& x, const char* what) {
    if (x.v.is_number_integer()) return x.v.get<std::int64_t>();
    if (x.v.is_boolean()) return x.v.get<bool>() ? 1 : 0;
    fail(std::string(what) + " must be an integer, got " + text_of(x));
}

double as_number(const Value& x, const char* what) {
    if (x.v.is_number() || x.v.is_boolean()) return x.v.get<double>();
    fail(std::string(what) + " must be a number, got " + text_of(x));
}

std::string strip(std::string s, bool left, bool right) {
    const auto space = [](unsigned char c) { return std::isspace(c) != 0; };
    if (right) {
        while (!s.empty() && space(static_cast<unsigned char>(s.back()))) s.pop_back();
    }
    if (left) {
        std::size_t i = 0;
        while (i < s.size() && space(static_cast<unsigned char>(s[i]))) ++i;
        s.erase(0, i);
    }
    return s;
}

// ------------------------------------------------------------ expressions

enum class Tok { Name, String, Int, Float, Op, End };

struct Token {
    Tok kind;
    std::string text;
};

std::vector<Token> lex_expression(std::string_view s) {
    std::vector<Token> out;
    std::size_t i = 0;
    while (i < s.size()) {
        const char c = s[i];
        if (std::isspace(static_cast<unsigned char>(c))) {
            ++i;
        } else if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            std::size_t j = i;
            while (j < s.size() && (std::isalnum(static_cast<unsigned char>(s[j])) || s[j] == '_'))
                ++j;
            out.push_back({Tok::Name, std::string(s.substr(i, j - i))});
            i = j;
        } else if (std::isdigit(static_cast<unsigned char>(c))) {
            std::size_t j = i;
            while (j < s.size() && std::isdigit(static_cast<unsigned char>(s[j]))) ++j;
            bool is_float = false;
            if (j + 1 < s.size() && s[j] == '.' &&
                std::isdigit(static_cast<unsigned char>(s[j + 1]))) {
                is_float = true;
                ++j;
                while (j < s.size() && std::isdigit(static_cast<unsigned char>(s[j]))) ++j;
            }
            out.push_back({is_float ? Tok::Float : Tok::Int, std::string(s.substr(i, j - i))});
            i = j;
        } else if (c == '\'' || c == '"') {
            std::string text;
            std::size_t j = i + 1;
            for (; j < s.size() && s[j] != c; ++j) {
                if (s[j] == '\\' && j + 1 < s.size()) {
                    const char e = s[++j];
                    text += e == 'n' ? '\n' : e == 't' ? '\t' : e == 'r' ? '\r' : e;
                } else {
                    text += s[j];
                }
            }
            if (j >= s.size()) fail("unterminated string in " + std::string(s));
            out.push_back({Tok::String, text});
            i = j + 1;
        } else {
            static const char* const two[] = {"==", "!=", "<=", ">=", "//"};
            const auto op =
                std::ranges::find_if(two, [&](const char* o) { return s.substr(i, 2) == o; });
            const std::size_t len = op != std::end(two) ? 2 : 1;
            if (len == 1 &&
                std::string_view("+-*/%~|.,:()[]{}<>=").find(c) == std::string_view::npos) {
                fail(std::string("unexpected character '") + c + "' in " + std::string(s));
            }
            out.push_back({Tok::Op, std::string(s.substr(i, len))});
            i += len;
        }
    }
    out.push_back({Tok::End, ""});
    return out;
}

}  // namespace

// Outside the anonymous namespace: Node, which the header names, holds these.
struct Expr {
    enum class Kind {
        Literal,
        Name,
        Attr,
        Index,
        Slice,
        Call,
        Filter,
        Unary,
        Binary,
        Test,
        Cond,
        List,
        Dict
    };
    Kind kind;
    Json literal;
    std::string name;  // variable, attribute, function, filter, test or operator
    std::vector<std::unique_ptr<Expr>> args;  // operands; null where optional and absent
    std::vector<std::string>
        kwarg_names;      // Call and Filter: keyword arguments, after the positional
    bool negate = false;  // Test: "is not"
};

using ExprPtr = std::unique_ptr<Expr>;

namespace {

ExprPtr make(Expr::Kind kind, std::string name = {}) {
    auto e = std::make_unique<Expr>();
    e->kind = kind;
    e->name = std::move(name);
    return e;
}

class ExprParser {
public:
    explicit ExprParser(std::string_view source)
        : source_(source), tokens_(lex_expression(source)) {}

    ExprPtr parse_all() {
        ExprPtr e = expression();
        expect_end();
        return e;
    }

    ExprPtr expression() { return conditional(); }
    // A for loop's iterable: an expression without an inline if, as in Jinja.
    ExprPtr iterable() { return or_expr(); }

    bool at_name(std::string_view word) const {
        return tokens_[pos_].kind == Tok::Name && tokens_[pos_].text == word;
    }
    bool at_op(std::string_view op) const {
        return tokens_[pos_].kind == Tok::Op && tokens_[pos_].text == op;
    }
    bool at_end() const { return tokens_[pos_].kind == Tok::End; }

    std::string name() {
        if (tokens_[pos_].kind != Tok::Name) error("a name");
        return tokens_[pos_++].text;
    }
    void expect_op(std::string_view op) {
        if (!at_op(op)) error("'" + std::string(op) + "'");
        ++pos_;
    }
    void expect_name(std::string_view word) {
        if (!at_name(word)) error("'" + std::string(word) + "'");
        ++pos_;
    }
    void expect_end() {
        if (!at_end()) error("the end of the expression");
    }

private:
    [[noreturn]] void error(const std::string& wanted) const {
        fail("expected " + wanted + " at '" + tokens_[pos_].text + "' in {" + std::string(source_) +
             "}");
    }

    ExprPtr binary(std::string op, ExprPtr l, ExprPtr r) {
        auto e = make(Expr::Kind::Binary, std::move(op));
        e->args.push_back(std::move(l));
        e->args.push_back(std::move(r));
        return e;
    }

    ExprPtr conditional() {
        ExprPtr value = or_expr();
        if (!at_name("if")) return value;
        ++pos_;
        auto e = make(Expr::Kind::Cond);
        e->args.push_back(std::move(value));
        e->args.push_back(or_expr());
        if (at_name("else")) {
            ++pos_;
            e->args.push_back(conditional());
        } else {
            e->args.push_back(nullptr);
        }
        return e;
    }

    ExprPtr or_expr() {
        ExprPtr l = and_expr();
        while (at_name("or")) {
            ++pos_;
            l = binary("or", std::move(l), and_expr());
        }
        return l;
    }

    ExprPtr and_expr() {
        ExprPtr l = not_expr();
        while (at_name("and")) {
            ++pos_;
            l = binary("and", std::move(l), not_expr());
        }
        return l;
    }

    ExprPtr not_expr() {
        if (!at_name("not")) return compare();
        ++pos_;
        auto e = make(Expr::Kind::Unary, "not");
        e->args.push_back(not_expr());
        return e;
    }

    ExprPtr compare() {
        ExprPtr l = concat();
        for (;;) {
            if (tokens_[pos_].kind == Tok::Op && (at_op("==") || at_op("!=") || at_op("<") ||
                                                  at_op("<=") || at_op(">") || at_op(">="))) {
                std::string op = tokens_[pos_++].text;
                l = binary(std::move(op), std::move(l), concat());
            } else if (at_name("in")) {
                ++pos_;
                l = binary("in", std::move(l), concat());
            } else if (at_name("not") && tokens_[pos_ + 1].kind == Tok::Name &&
                       tokens_[pos_ + 1].text == "in") {
                pos_ += 2;
                l = binary("not in", std::move(l), concat());
            } else if (at_name("is")) {
                ++pos_;
                auto e = make(Expr::Kind::Test);
                if (at_name("not")) {
                    ++pos_;
                    e->negate = true;
                }
                e->name = name();
                e->args.push_back(std::move(l));
                l = std::move(e);
            } else {
                return l;
            }
        }
    }

    ExprPtr concat() {
        ExprPtr l = additive();
        while (at_op("~")) {
            ++pos_;
            l = binary("~", std::move(l), additive());
        }
        return l;
    }

    ExprPtr additive() {
        ExprPtr l = multiplicative();
        while (at_op("+") || at_op("-")) {
            std::string op = tokens_[pos_++].text;
            l = binary(std::move(op), std::move(l), multiplicative());
        }
        return l;
    }

    ExprPtr multiplicative() {
        ExprPtr l = unary();
        while (at_op("*") || at_op("/") || at_op("//") || at_op("%")) {
            std::string op = tokens_[pos_++].text;
            l = binary(std::move(op), std::move(l), unary());
        }
        return l;
    }

    ExprPtr unary() {
        if (at_op("-")) {
            ++pos_;
            auto e = make(Expr::Kind::Unary, "-");
            e->args.push_back(unary());
            return e;
        }
        if (at_op("+")) {
            ++pos_;
            return unary();
        }
        return filtered();
    }

    // Filters bind tighter than arithmetic: a + b | trim is a + (b | trim).
    ExprPtr filtered() {
        ExprPtr e = postfix();
        while (at_op("|")) {
            ++pos_;
            auto f = make(Expr::Kind::Filter, name());
            f->args.push_back(std::move(e));
            if (at_op("(")) call_arguments(*f);
            e = std::move(f);
        }
        return e;
    }

    void call_arguments(Expr& call) {
        expect_op("(");
        while (!at_op(")")) {
            if (tokens_[pos_].kind == Tok::Name && tokens_[pos_ + 1].kind == Tok::Op &&
                tokens_[pos_ + 1].text == "=") {
                call.kwarg_names.push_back(name());
                ++pos_;
            } else if (!call.kwarg_names.empty()) {
                error("a keyword argument");
            }
            call.args.push_back(expression());
            if (!at_op(")")) expect_op(",");
        }
        ++pos_;
    }

    ExprPtr postfix() {
        ExprPtr e = primary();
        for (;;) {
            if (at_op(".")) {
                ++pos_;
                auto a = make(Expr::Kind::Attr, name());
                a->args.push_back(std::move(e));
                e = std::move(a);
            } else if (at_op("[")) {
                ++pos_;
                ExprPtr first = at_op(":") ? nullptr : expression();
                if (at_op(":")) {
                    ++pos_;
                    auto s = make(Expr::Kind::Slice);
                    s->args.push_back(std::move(e));
                    s->args.push_back(std::move(first));
                    s->args.push_back(at_op("]") ? nullptr : expression());
                    e = std::move(s);
                } else {
                    auto i = make(Expr::Kind::Index);
                    i->args.push_back(std::move(e));
                    i->args.push_back(std::move(first));
                    e = std::move(i);
                }
                expect_op("]");
            } else if (at_op("(")) {
                auto c = make(Expr::Kind::Call);
                c->args.push_back(std::move(e));
                call_arguments(*c);
                e = std::move(c);
            } else {
                return e;
            }
        }
    }

    ExprPtr primary() {
        const Token t = tokens_[pos_];
        if (t.kind == Tok::String || t.kind == Tok::Int || t.kind == Tok::Float) {
            ++pos_;
            auto e = make(Expr::Kind::Literal);
            if (t.kind == Tok::String) {
                e->literal = t.text;
                while (tokens_[pos_].kind == Tok::String)
                    e->literal = e->literal.get<std::string>() + tokens_[pos_++].text;
            } else if (t.kind == Tok::Int) {
                e->literal = std::stoll(t.text);
            } else {
                e->literal = std::stod(t.text);
            }
            return e;
        }
        if (t.kind == Tok::Name) {
            ++pos_;
            auto e = make(Expr::Kind::Literal);
            if (t.text == "true" || t.text == "True") {
                e->literal = true;
            } else if (t.text == "false" || t.text == "False") {
                e->literal = false;
            } else if (t.text == "none" || t.text == "None") {
                e->literal = nullptr;
            } else {
                return make(Expr::Kind::Name, t.text);
            }
            return e;
        }
        if (at_op("(")) {
            ++pos_;
            ExprPtr e = expression();
            expect_op(")");
            return e;
        }
        if (at_op("[")) {
            ++pos_;
            auto e = make(Expr::Kind::List);
            while (!at_op("]")) {
                e->args.push_back(expression());
                if (!at_op("]")) expect_op(",");
            }
            ++pos_;
            return e;
        }
        if (at_op("{")) {
            ++pos_;
            auto e = make(Expr::Kind::Dict);
            while (!at_op("}")) {
                e->args.push_back(expression());
                expect_op(":");
                e->args.push_back(expression());
                if (!at_op("}")) expect_op(",");
            }
            ++pos_;
            return e;
        }
        error("a value");
    }

    std::string_view source_;
    std::vector<Token> tokens_;
    std::size_t pos_ = 0;
};

// -------------------------------------------------------- template pieces

struct Piece {
    enum class Kind { Text, Output, Block, Comment } kind;
    std::string text;
    bool strip_before = false;  // {{- {%- {#-
    bool strip_after = false;   // -}} -%} -#}
    bool keep_indent = false;   // {%+ : no lstrip_blocks
};

// Splits a template into text and tags, applying Jinja's whitespace control
// and the trim_blocks and lstrip_blocks settings transformers uses.
std::vector<Piece> lex_template(std::string_view source) {
    // Jinja drops a single trailing newline (keep_trailing_newline is off).
    if (source.ends_with("\r\n")) {
        source.remove_suffix(2);
    } else if (source.ends_with('\n')) {
        source.remove_suffix(1);
    }

    std::vector<Piece> pieces;
    std::size_t pos = 0;
    while (pos < source.size()) {
        std::size_t open = std::string_view::npos;
        for (const char* d : {"{{", "{%", "{#"}) open = std::min(open, source.find(d, pos));
        if (open == std::string_view::npos) {
            pieces.push_back({Piece::Kind::Text, std::string(source.substr(pos))});
            break;
        }
        if (open > pos)
            pieces.push_back({Piece::Kind::Text, std::string(source.substr(pos, open - pos))});

        Piece tag;
        const char type = source[open + 1];
        tag.kind = type == '{'   ? Piece::Kind::Output
                   : type == '%' ? Piece::Kind::Block
                                 : Piece::Kind::Comment;
        std::size_t start = open + 2;
        if (start < source.size() && source[start] == '-') {
            tag.strip_before = true;
            ++start;
        } else if (start < source.size() && source[start] == '+') {
            tag.keep_indent = true;
            ++start;
        }
        const std::string_view close = type == '{' ? "}}" : type == '%' ? "%}" : "#}";
        // Find the close outside string literals.
        std::size_t end = start;
        char quote = 0;
        for (; end < source.size(); ++end) {
            const char c = source[end];
            if (quote) {
                if (c == '\\') {
                    ++end;
                } else if (c == quote) {
                    quote = 0;
                }
            } else if (type != '#' && (c == '\'' || c == '"')) {
                quote = c;
            } else if (source.substr(end, 2) == close) {
                break;
            }
        }
        if (end >= source.size())
            fail("unclosed tag starting " + std::string(source.substr(open, 20)));
        std::size_t content_end = end;
        if (content_end > start && source[content_end - 1] == '-') {
            tag.strip_after = true;
            --content_end;
        }
        tag.text = std::string(source.substr(start, content_end - start));
        pieces.push_back(std::move(tag));
        pos = end + 2;
    }

    // Whitespace control, looking at each tag's neighbouring text.
    for (std::size_t i = 0; i < pieces.size(); ++i) {
        Piece& tag = pieces[i];
        if (tag.kind == Piece::Kind::Text) continue;
        const bool block = tag.kind != Piece::Kind::Output;
        Piece* before = i > 0 && pieces[i - 1].kind == Piece::Kind::Text ? &pieces[i - 1] : nullptr;
        Piece* after = i + 1 < pieces.size() && pieces[i + 1].kind == Piece::Kind::Text
                           ? &pieces[i + 1]
                           : nullptr;

        if (before && tag.strip_before) {
            before->text = strip(before->text, false, true);
        } else if (before && block && !tag.keep_indent) {
            // lstrip_blocks: drop spaces and tabs between the line start and the tag.
            const std::size_t line = before->text.find_last_of('\n');
            const std::size_t from = line == std::string::npos ? 0 : line + 1;
            const bool line_start = line != std::string::npos || i == 1;
            if (line_start && before->text.find_first_not_of(" \t", from) == std::string::npos) {
                before->text.erase(from);
            }
        }
        if (after && tag.strip_after) {
            after->text = strip(after->text, true, false);
        } else if (after && block) {
            // trim_blocks: drop the first newline after a block tag.
            if (after->text.starts_with("\r\n")) {
                after->text.erase(0, 2);
            } else if (after->text.starts_with('\n')) {
                after->text.erase(0, 1);
            }
        }
    }
    std::erase_if(pieces, [](const Piece& p) { return p.kind == Piece::Kind::Comment; });
    return pieces;
}

}  // namespace

// ------------------------------------------------------------- statements

struct Node {
    enum class Kind { Text, Output, If, For, Set, Break, Continue, Sequence };
    Kind kind;
    std::string text;                 // Text; Set target
    std::string attr;                 // Set: the attribute of the target, for ns.attr
    ExprPtr expr;                     // Output value; For iterable; Set value
    ExprPtr loop_filter;              // For: the condition after the iterable, if any
    std::vector<std::string> vars;    // For targets
    std::vector<ExprPtr> conditions;  // If: one per branch
    std::vector<std::vector<std::unique_ptr<Node>>> bodies;  // If branches; For body; Sequence
    std::vector<std::unique_ptr<Node>> else_body;            // If and For
};

namespace {

using Nodes = std::vector<std::unique_ptr<Node>>;

std::unique_ptr<Node> make_node(Node::Kind kind) {
    auto n = std::make_unique<Node>();
    n->kind = kind;
    return n;
}

std::string first_word(const std::string& text) {
    const std::size_t b = text.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return {};
    std::size_t e = b;
    while (e < text.size() && (std::isalnum(static_cast<unsigned char>(text[e])) || text[e] == '_'))
        ++e;
    return text.substr(b, e - b);
}

std::string after_first_word(const std::string& text) {
    const std::string word = first_word(text);
    return text.substr(text.find(word) + word.size());
}

class TemplateParser {
public:
    explicit TemplateParser(std::vector<Piece> pieces) : pieces_(std::move(pieces)) {}

    Nodes parse() {
        Nodes nodes = parse_until({});
        if (pos_ < pieces_.size()) fail("unexpected {% " + pieces_[pos_].text + " %}");
        return nodes;
    }

private:
    // Parses nodes until a block tag whose keyword is in ends (left unconsumed).
    Nodes parse_until(std::initializer_list<std::string_view> ends) {
        Nodes nodes;
        while (pos_ < pieces_.size()) {
            const Piece& p = pieces_[pos_];
            if (p.kind == Piece::Kind::Text) {
                auto n = make_node(Node::Kind::Text);
                n->text = p.text;
                nodes.push_back(std::move(n));
                ++pos_;
                continue;
            }
            if (p.kind == Piece::Kind::Output) {
                auto n = make_node(Node::Kind::Output);
                n->expr = ExprParser(p.text).parse_all();
                nodes.push_back(std::move(n));
                ++pos_;
                continue;
            }
            const std::string word = first_word(p.text);
            if (std::ranges::find(ends, word) != ends.end()) return nodes;
            ++pos_;
            nodes.push_back(statement(word, after_first_word(p.text)));
        }
        if (ends.size() != 0) {
            std::string expected;
            for (const std::string_view end : ends) {
                expected += (expected.empty() ? "" : " or ") + std::string("{% ") +
                            std::string(end) + " %}";
            }
            fail("the template ends where it needs " + expected);
        }
        return nodes;
    }

    std::string take_block() { return pieces_[pos_++].text; }

    std::unique_ptr<Node> statement(const std::string& word, const std::string& rest) {
        if (word == "if") {
            auto n = make_node(Node::Kind::If);
            n->conditions.push_back(ExprParser(rest).parse_all());
            n->bodies.push_back(parse_until({"elif", "else", "endif"}));
            for (;;) {
                const std::string tag = take_block();
                const std::string kw = first_word(tag);
                if (kw == "elif") {
                    n->conditions.push_back(ExprParser(after_first_word(tag)).parse_all());
                    n->bodies.push_back(parse_until({"elif", "else", "endif"}));
                } else if (kw == "else") {
                    n->else_body = parse_until({"endif"});
                    take_block();
                    return n;
                } else {
                    return n;  // endif
                }
            }
        }
        if (word == "for") {
            auto n = make_node(Node::Kind::For);
            ExprParser p(rest);
            n->vars.push_back(p.name());
            while (p.at_op(",")) {
                p.expect_op(",");
                n->vars.push_back(p.name());
            }
            p.expect_name("in");
            // As Jinja: the iterable has no inline if, so a trailing one filters.
            n->expr = p.iterable();
            if (p.at_name("if")) {
                p.expect_name("if");
                n->loop_filter = p.expression();
            }
            if (!p.at_end()) fail("recursive for loops are not supported: {% for" + rest + " %}");
            n->bodies.push_back(parse_until({"else", "endfor"}));
            if (first_word(take_block()) == "else") {
                n->else_body = parse_until({"endfor"});
                take_block();
            }
            return n;
        }
        if (word == "set") {
            auto n = make_node(Node::Kind::Set);
            ExprParser p(rest);
            n->text = p.name();
            if (p.at_op(".")) {
                p.expect_op(".");
                n->attr = p.name();
            }
            if (!p.at_op("="))
                fail("block set and multiple targets are not supported: {% set" + rest + " %}");
            p.expect_op("=");
            n->expr = p.expression();
            p.expect_end();
            return n;
        }
        if (word == "break") return make_node(Node::Kind::Break);
        if (word == "continue") return make_node(Node::Kind::Continue);
        if (word == "generation") {  // transformers marks assistant text with it; it renders as is
            auto n = make_node(Node::Kind::Sequence);
            n->bodies.push_back(parse_until({"endgeneration"}));
            take_block();
            return n;
        }
        fail("the tag {% " + word + " %} is not supported");
    }

    std::vector<Piece> pieces_;
    std::size_t pos_ = 0;
};

// ------------------------------------------------------------- evaluation

enum class Flow { Normal, Break, Continue };

class Renderer {
public:
    explicit Renderer(const Json& context) {
        scopes_.emplace_back();
        for (const auto& [k, v] : context.items()) scopes_.back()[k] = Value{v};
    }

    std::string rendered;

    Flow run(const Nodes& nodes) {
        for (const auto& node : nodes) {
            if (const Flow f = run(*node); f != Flow::Normal) return f;
        }
        return Flow::Normal;
    }

private:
    Flow run(const Node& n) {
        switch (n.kind) {
            case Node::Kind::Text: rendered += n.text; return Flow::Normal;
            case Node::Kind::Output: rendered += text_of(eval(*n.expr)); return Flow::Normal;
            case Node::Kind::Set:
                if (n.attr.empty()) {
                    scopes_.back()[n.text] = eval(*n.expr);
                } else {
                    set_attribute(n.text, n.attr, eval(*n.expr));
                }
                return Flow::Normal;
            case Node::Kind::Break: return Flow::Break;
            case Node::Kind::Continue: return Flow::Continue;
            case Node::Kind::Sequence: return run(n.bodies[0]);
            case Node::Kind::If:
                for (std::size_t i = 0; i < n.conditions.size(); ++i) {
                    if (truthy(eval(*n.conditions[i]))) return run(n.bodies[i]);
                }
                return run(n.else_body);
            case Node::Kind::For: return run_for(n);
        }
        return Flow::Normal;
    }

    // Binds a for loop's targets to item: the item, or its elements unpacked.
    static void bind_loop_vars(std::map<std::string, Value>& scope, const Node& n,
                               const Json& item) {
        if (n.vars.size() == 1) {
            scope[n.vars[0]] = Value{item};
            return;
        }
        if (!item.is_array() || item.size() != n.vars.size())
            fail("cannot unpack " + text_of(item, true));
        for (std::size_t k = 0; k < n.vars.size(); ++k) scope[n.vars[k]] = Value{item[k]};
    }

    Flow run_for(const Node& n) {
        const Value iterable = eval(*n.expr);
        Json items = Json::array();
        if (iterable.v.is_array()) {
            items = iterable.v;
        } else if (iterable.v.is_object()) {
            for (const auto& [k, v] : iterable.v.items())
                items.push_back(k);  // iterating a dict gives keys
        } else if (iterable.v.is_string()) {
            for (const char c : iterable.v.get<std::string>()) items.push_back(std::string(1, c));
        } else if (!iterable.undefined && !iterable.v.is_null()) {
            fail("cannot loop over " + text_of(iterable));
        }
        // {% for x in xs if cond %}: the loop, loop.index included, runs over
        // the items that pass.
        if (n.loop_filter) {
            Json kept = Json::array();
            for (const Json& item : items) {
                bind_loop_vars(scopes_.emplace_back(), n, item);
                const bool pass = truthy(eval(*n.loop_filter));
                scopes_.pop_back();
                if (pass) kept.push_back(item);
            }
            items = std::move(kept);
        }
        if (items.empty()) return run(n.else_body);

        const auto length = static_cast<std::int64_t>(items.size());
        for (std::int64_t i = 0; i < length; ++i) {
            auto& scope = scopes_.emplace_back();
            bind_loop_vars(scope, n, items[std::size_t(i)]);
            scope["loop"] = Value{Json{{"index", i + 1},
                                       {"index0", i},
                                       {"revindex", length - i},
                                       {"revindex0", length - i - 1},
                                       {"first", i == 0},
                                       {"last", i == length - 1},
                                       {"length", length}}};
            const Flow f = run(n.bodies[0]);
            scopes_.pop_back();
            if (f == Flow::Break) break;
        }
        return Flow::Normal;
    }

    // {% set name.attr = value %}: namespaces are objects that sets inside
    // loops can change, where a plain set only binds a name in its scope.
    void set_attribute(const std::string& name, const std::string& attr, const Value& value) {
        for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
            if (const auto found = it->find(name); found != it->end()) {
                if (!found->second.is_namespace) break;
                found->second.v[attr] = value.v;
                return;
            }
        }
        fail("cannot assign attribute on non-namespace object: " + name);
    }

    Value lookup(const std::string& name) const {
        for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
            if (const auto found = it->find(name); found != it->end()) return found->second;
        }
        // The functions call() provides are defined names too: LLaMA 3's
        // templates print today's date only if "strftime_now is defined".
        if (name == "raise_exception" || name == "range" || name == "namespace" ||
            name == "strftime_now") {
            return {Json("<built-in function " + name + ">")};
        }
        return undefined();
    }

    std::vector<Value> eval_args(const Expr& e, std::size_t skip) {
        std::vector<Value> out;
        for (std::size_t i = skip; i < e.args.size(); ++i) out.push_back(eval(*e.args[i]));
        return out;
    }

    Value eval(const Expr& e) {
        using K = Expr::Kind;
        switch (e.kind) {
            case K::Literal: return {e.literal};
            case K::Name: return lookup(e.name);
            case K::Attr: {
                const Value base = eval(*e.args[0]);
                if (base.v.is_object() && base.v.contains(e.name)) return {base.v[e.name]};
                return undefined();
            }
            case K::Index: return index(eval(*e.args[0]), eval(*e.args[1]));
            case K::Slice: return slice(e);
            case K::Call: return call(e);
            case K::Filter: return filter(e);
            case K::Unary: {
                const Value x = eval(*e.args[0]);
                if (e.name == "not") return {!truthy(x)};
                if (x.v.is_number_integer()) return {-x.v.get<std::int64_t>()};
                return {-as_number(x, "operand of -")};
            }
            case K::Binary: return binary(e);
            case K::Test: return {test(e.name, eval(*e.args[0])) != e.negate};
            case K::Cond:
                if (truthy(eval(*e.args[1]))) return eval(*e.args[0]);
                return e.args[2] ? eval(*e.args[2]) : undefined();
            case K::List: {
                Json list = Json::array();
                for (const auto& a : e.args) list.push_back(eval(*a).v);
                return {list};
            }
            case K::Dict: {
                Json dict = Json::object();
                for (std::size_t i = 0; i < e.args.size(); i += 2) {
                    dict[text_of(eval(*e.args[i]))] = eval(*e.args[i + 1]).v;
                }
                return {dict};
            }
        }
        return undefined();
    }

    static Value index(const Value& base, const Value& key) {
        if (base.v.is_object() && key.v.is_string()) {
            const auto& k = key.v.get_ref<const std::string&>();
            return base.v.contains(k) ? Value{base.v[k]} : undefined();
        }
        if ((base.v.is_array() || base.v.is_string()) && key.v.is_number_integer()) {
            const std::int64_t size =
                base.v.is_array() ? std::int64_t(base.v.size())
                                  : std::int64_t(base.v.get_ref<const std::string&>().size());
            std::int64_t i = key.v.get<std::int64_t>();
            if (i < 0) i += size;
            if (i < 0 || i >= size) return undefined();
            if (base.v.is_array()) return {base.v[std::size_t(i)]};
            return {std::string(1, base.v.get_ref<const std::string&>()[std::size_t(i)])};
        }
        return undefined();
    }

    Value slice(const Expr& e) {
        const Value base = eval(*e.args[0]);
        const bool is_string = base.v.is_string();
        if (!base.v.is_array() && !is_string) fail("cannot slice " + text_of(base));
        const std::int64_t size = is_string
                                      ? std::int64_t(base.v.get_ref<const std::string&>().size())
                                      : std::int64_t(base.v.size());
        const auto bound = [&](const ExprPtr& x, std::int64_t fallback) {
            if (!x) return fallback;
            std::int64_t v = as_int(eval(*x), "slice bound");
            if (v < 0) v += size;
            return std::clamp<std::int64_t>(v, 0, size);
        };
        const std::int64_t lo = bound(e.args[1], 0), hi = std::max(lo, bound(e.args[2], size));
        if (is_string)
            return {base.v.get<std::string>().substr(std::size_t(lo), std::size_t(hi - lo))};
        Json out = Json::array();
        for (std::int64_t i = lo; i < hi; ++i) out.push_back(base.v[std::size_t(i)]);
        return {out};
    }

    Value call(const Expr& e) {
        const Expr& callee = *e.args[0];
        std::vector<Value> args = eval_args(e, 1);
        if (callee.kind == Expr::Kind::Attr)
            return method(eval(*callee.args[0]), callee.name, args);
        if (callee.kind != Expr::Kind::Name) fail("only named functions and methods can be called");
        if (callee.name == "raise_exception") {
            // Not "chat template: ...": the template's own message, as transformers reports it.
            throw Error(args.empty() ? "raise_exception" : text_of(args[0]));
        }
        if (callee.name == "namespace") {
            // namespace(a=1, b=2): the keyword arguments are the last args.
            if (args.size() != e.kwarg_names.size()) {
                fail("namespace takes keyword arguments only");
            }
            Json object = Json::object();
            for (std::size_t i = 0; i < args.size(); ++i) object[e.kwarg_names[i]] = args[i].v;
            return {object, false, true};
        }
        if (callee.name == "range") {
            if (args.empty() || args.size() > 2) fail("range takes 1 or 2 arguments");
            const std::int64_t lo = args.size() == 2 ? as_int(args[0], "range start") : 0;
            const std::int64_t hi = as_int(args.back(), "range stop");
            Json out = Json::array();
            for (std::int64_t i = lo; i < hi; ++i) out.push_back(i);
            return {out};
        }
        if (callee.name == "strftime_now") {
            const std::time_t now = std::time(nullptr);
            std::tm local{};
#ifdef _WIN32
            localtime_s(&local, &now);
#else
            localtime_r(&now, &local);
#endif
            char buf[128];
            const std::string format = args.empty() ? "%Y-%m-%d" : text_of(args[0]);
            return {std::string(buf, std::strftime(buf, sizeof buf, format.c_str(), &local))};
        }
        fail("the function " + callee.name + " is not supported");
    }

    static Value method(const Value& self, const std::string& name,
                        const std::vector<Value>& args) {
        if (self.v.is_string()) {
            const std::string s = self.v.get<std::string>();
            const auto arg = [&](std::size_t i) {
                return i < args.size() ? text_of(args[i]) : std::string();
            };
            if (name == "strip") return {strip(s, true, true)};
            if (name == "lstrip") return {strip(s, true, false)};
            if (name == "rstrip") return {strip(s, false, true)};
            if (name == "upper" || name == "lower") {
                std::string out = s;
                for (char& c : out) {
                    c = static_cast<char>(name == "upper"
                                              ? std::toupper(static_cast<unsigned char>(c))
                                              : std::tolower(static_cast<unsigned char>(c)));
                }
                return {out};
            }
            if (name == "title") {
                std::string out = s;
                bool start = true;
                for (char& c : out) {
                    const auto u = static_cast<unsigned char>(c);
                    c = static_cast<char>(start ? std::toupper(u) : std::tolower(u));
                    start = !std::isalpha(u);
                }
                return {out};
            }
            if (name == "startswith") return {s.starts_with(arg(0))};
            if (name == "endswith") return {s.ends_with(arg(0))};
            if (name == "replace") {
                std::string out = s;
                const std::string from = arg(0), to = arg(1);
                if (!from.empty()) {
                    for (std::size_t at = out.find(from); at != std::string::npos;
                         at = out.find(from, at + to.size())) {
                        out.replace(at, from.size(), to);
                    }
                }
                return {out};
            }
        }
        if (self.v.is_object()) {
            if (name == "items") {
                Json out = Json::array();
                for (const auto& [k, v] : self.v.items()) out.push_back(Json::array({k, v}));
                return {out};
            }
            if (name == "keys" || name == "values") {
                Json out = Json::array();
                for (const auto& [k, v] : self.v.items())
                    out.push_back(name == "keys" ? Json(k) : v);
                return {out};
            }
            if (name == "get") {
                const std::string key = args.empty() ? std::string() : text_of(args[0]);
                if (self.v.contains(key)) return {self.v[key]};
                return args.size() > 1 ? args[1] : Value{Json(nullptr)};
            }
        }
        fail("the method " + name + " is not supported on " + text_of(self.v, true));
    }

    Value filter(const Expr& e) {
        const Value x = eval(*e.args[0]);
        const std::vector<Value> args = eval_args(e, 1);
        const std::string& f = e.name;
        if (f == "trim") return {strip(text_of(x), true, true)};
        if (f == "upper" || f == "lower") return method(Value{text_of(x)}, f, {});
        if (f == "string") return {text_of(x)};
        if (f == "int")
            return {x.v.is_string() ? std::stoll(x.v.get<std::string>())
                                    : as_int(x, "int filter input")};
        if (f == "length" || f == "count") {
            if (x.v.is_string()) return {std::int64_t(x.v.get_ref<const std::string&>().size())};
            if (x.v.is_array() || x.v.is_object()) return {std::int64_t(x.v.size())};
            return {std::int64_t{0}};
        }
        if (f == "first" || f == "last") {
            if (!x.v.is_array() || x.v.empty()) return undefined();
            return {f == "first" ? x.v.front() : x.v.back()};
        }
        if (f == "join") {
            const std::string sep = args.empty() ? "" : text_of(args[0]);
            std::string out;
            if (x.v.is_array()) {
                for (std::size_t i = 0; i < x.v.size(); ++i)
                    out += (i ? sep : "") + text_of(x.v[i], false);
            }
            return {out};
        }
        if (f == "default" || f == "d") {
            const bool falsy_too = args.size() > 1 && truthy(args[1]);
            if (x.undefined || (falsy_too && !truthy(x)))
                return args.empty() ? Value{Json("")} : args[0];
            return x;
        }
        if (f == "tojson") return {to_json(x.v)};
        if (f == "list") {
            Json out = Json::array();
            if (x.v.is_array()) {
                out = x.v;
            } else if (x.v.is_string()) {
                for (const char c : x.v.get<std::string>()) out.push_back(std::string(1, c));
            } else if (x.v.is_object()) {
                for (const auto& [k, v] : x.v.items()) out.push_back(k);
            }
            return {out};
        }
        if (f == "selectattr" || f == "rejectattr") {
            // selectattr(attr) keeps items whose attr is truthy; selectattr(attr,
            // test, arg...) those whose attr passes the test.
            if (args.empty()) fail(f + " needs an attribute name");
            const std::string attr = text_of(args[0]);
            Json out = Json::array();
            if (x.v.is_array()) {
                for (const Json& item : x.v) {
                    const Value a =
                        item.is_object() && item.contains(attr) ? Value{item[attr]} : undefined();
                    bool pass = truthy(a);
                    if (args.size() > 1) {
                        const std::string t = text_of(args[1]);
                        if (t == "equalto" || t == "eq" || t == "==" || t == "ne" || t == "!=") {
                            if (args.size() < 3) fail(f + ": the test " + t + " needs a value");
                            const bool equal = !a.undefined && a.v == args[2].v;
                            pass = equal == (t == "equalto" || t == "eq" || t == "==");
                        } else {
                            pass = test(t, a);
                        }
                    }
                    if (pass == (f == "selectattr")) out.push_back(item);
                }
            }
            return {out};
        }
        fail("the filter " + f + " is not supported");
    }

    static bool test(const std::string& name, const Value& x) {
        if (name == "defined") return !x.undefined;
        if (name == "undefined") return x.undefined;
        if (name == "none") return !x.undefined && x.v.is_null();
        if (name == "string") return x.v.is_string();
        if (name == "number") return x.v.is_number();
        if (name == "integer") return x.v.is_number_integer();
        if (name == "boolean") return x.v.is_boolean();
        if (name == "mapping") return x.v.is_object();
        if (name == "sequence" || name == "iterable")
            return x.v.is_array() || x.v.is_string() || x.v.is_object();
        if (name == "true") return x.v.is_boolean() && x.v.get<bool>();
        if (name == "false") return x.v.is_boolean() && !x.v.get<bool>();
        if (name == "odd" || name == "even")
            return (as_int(x, "odd/even test input") % 2 != 0) == (name == "odd");
        fail("the test " + name + " is not supported");
    }

    Value binary(const Expr& e) {
        const std::string& op = e.name;
        if (op == "and" || op == "or") {  // Python semantics: return an operand
            const Value l = eval(*e.args[0]);
            if ((op == "and") != truthy(l)) return l;
            return eval(*e.args[1]);
        }
        const Value l = eval(*e.args[0]), r = eval(*e.args[1]);
        if (op == "~") return {text_of(l) + text_of(r)};
        if (op == "==" || op == "!=") {
            const bool equal =
                (l.undefined && r.undefined) || (!l.undefined && !r.undefined && l.v == r.v);
            return {equal == (op == "==")};
        }
        if (op == "in" || op == "not in") {
            bool found = false;
            if (r.v.is_string()) {
                found = r.v.get_ref<const std::string&>().find(text_of(l)) != std::string::npos;
            } else if (r.v.is_array()) {
                found = std::ranges::find(r.v, l.v) != r.v.end();
            } else if (r.v.is_object()) {
                found = l.v.is_string() && r.v.contains(l.v.get<std::string>());
            }
            return {found == (op == "in")};
        }
        if (op == "<" || op == "<=" || op == ">" || op == ">=") {
            int cmp;
            if (l.v.is_string() && r.v.is_string()) {
                cmp = l.v.get<std::string>().compare(r.v.get<std::string>());
            } else {
                const double a = as_number(l, "comparison operand"),
                             b = as_number(r, "comparison operand");
                cmp = a < b ? -1 : a > b ? 1 : 0;
            }
            return {op == "<" ? cmp < 0 : op == "<=" ? cmp <= 0 : op == ">" ? cmp > 0 : cmp >= 0};
        }
        if (op == "+" && l.v.is_string() && r.v.is_string())
            return {l.v.get<std::string>() + r.v.get<std::string>()};
        if (op == "+" && l.v.is_array() && r.v.is_array()) {
            Json out = l.v;
            for (const auto& x : r.v) out.push_back(x);
            return {out};
        }
        const bool ints = (l.v.is_number_integer() || l.v.is_boolean()) &&
                          (r.v.is_number_integer() || r.v.is_boolean());
        if (ints && op != "/") {
            const std::int64_t a = as_int(l, "operand"), b = as_int(r, "operand");
            if ((op == "//" || op == "%") && b == 0) fail("division by zero");
            if (op == "+") return {a + b};
            if (op == "-") return {a - b};
            if (op == "*") return {a * b};
            // Python floors towards negative infinity.
            const std::int64_t q = a / b - ((a % b != 0) && ((a < 0) != (b < 0)) ? 1 : 0);
            return {op == "//" ? q : a - q * b};
        }
        const double a = as_number(l, "operand"), b = as_number(r, "operand");
        if (op == "+") return {a + b};
        if (op == "-") return {a - b};
        if (op == "*") return {a * b};
        if (b == 0) fail("division by zero");
        if (op == "/") return {a / b};
        if (op == "//") return {std::floor(a / b)};
        return {a - std::floor(a / b) * b};
    }

    std::vector<std::map<std::string, Value>> scopes_;
};

}  // namespace

Template::Template(std::string_view source)
    : nodes_(TemplateParser(lex_template(source)).parse()) {}
Template::~Template() = default;
Template::Template(Template&&) noexcept = default;
Template& Template::operator=(Template&&) noexcept = default;

std::string Template::render(const nlohmann::ordered_json& context) const {
    Renderer r{context};
    r.run(nodes_);
    return std::move(r.rendered);
}

}  // namespace vkml::detail::jinja
