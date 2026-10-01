#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <vkml/vkml.hpp>

#include "io/gguf.hpp"
#include "support/gguf_writer.hpp"
#include "support/safetensors_writer.hpp"

using Catch::Matchers::ContainsSubstring;
using vkml::DType;
using vkml::Shape;
using vkml::detail::Gguf;
using namespace vkml_test;

namespace {

// The bits of a 16-bit tensor's elements.
std::vector<std::uint16_t> half_bits(const vkml::Tensor& t) {
    const std::vector<std::byte> bytes = t.to_bytes();
    std::vector<std::uint16_t> out(bytes.size() / 2);
    std::memcpy(out.data(), bytes.data(), out.size() * 2);
    return out;
}

template <class T>
void append(std::vector<std::uint8_t>& out, T v) {
    const auto* p = reinterpret_cast<const std::uint8_t*>(&v);
    out.insert(out.end(), p, p + sizeof(T));
}

// One Q8_0 block: an f16 scale, then 32 int8 values.
void q8_block(std::vector<std::uint8_t>& out, std::uint16_t scale_f16,
              const std::vector<std::int8_t>& q) {
    append(out, scale_f16);
    for (const std::int8_t v : q) append(out, v);
}

}  // namespace

TEST_CASE("Gguf reads metadata of every type and the tensor table", "[gguf]") {
    GgufWriter w;
    w.string("general.architecture", "llama");
    w.u32("llama.block_count", 2);
    w.i32("llama.some_signed", -7);
    w.u64("llama.context_length", 4096);
    w.f32("llama.rope.freq_base", 10000.0f);
    w.boolean("tokenizer.ggml.add_bos_token", true);
    w.strings("tokenizer.ggml.tokens", {"<s>", "a", "b"});
    w.i32s("tokenizer.ggml.token_type", {3, 1, 1});
    w.tensor("w", {4, 3}, kGgmlF32, raw(std::vector<float>(12, 1.0f)));
    const auto path = temp_path("vkml_meta.gguf");
    w.write(path);

    const Gguf g{path};
    CHECK(g.version() == 3);
    const auto& m = g.metadata();
    CHECK(m.at("general.architecture") == "llama");
    CHECK(m.at("llama.block_count") == 2);
    CHECK(m.at("llama.some_signed") == -7);
    CHECK(m.at("llama.context_length") == 4096);
    CHECK(m.at("llama.rope.freq_base") == 10000.0);
    CHECK(m.at("tokenizer.ggml.add_bos_token") == true);
    CHECK(m.at("tokenizer.ggml.tokens") == nlohmann::json::array({"<s>", "a", "b"}));
    CHECK(m.at("tokenizer.ggml.token_type") == nlohmann::json::array({3, 1, 1}));
    CHECK(g.names() == std::vector<std::string>{"w"});
    CHECK(g.contains("w"));
    CHECK(g.shape("w") == Shape{3, 4});  // row-major: GGUF lists the row length first
    CHECK(g.type_name("w") == "f32");
}

TEST_CASE("Gguf loads float tensors as they are stored", "[gguf]") {
    GgufWriter w;
    std::vector<float> values(12);
    for (std::size_t i = 0; i < values.size(); ++i) values[i] = float(i) - 5.5f;
    w.tensor("a", {4, 3}, kGgmlF32, raw(values));
    // f16 1, 2, -0.5, 65504: 0x3C00, 0x4000, 0xB800, 0x7BFF
    w.tensor("b", {2, 2}, kGgmlF16,
             raw(std::vector<std::uint16_t>{0x3C00, 0x4000, 0xB800, 0x7BFF}));
    w.tensor("c", {3}, kGgmlF32, raw(std::vector<float>{1, 2, 3}));
    const auto path = temp_path("vkml_floats.gguf");
    w.write(path);

    vkml::Context context;
    const Gguf g{path};
    const vkml::Tensor a = g.load(context, "a");
    CHECK(a.shape() == Shape{3, 4});
    CHECK(a.to_vector<float>() == values);
    const vkml::Tensor b = g.load(context, "b");
    CHECK(b.dtype() == DType::F16);
    CHECK(vkml::cast(b, DType::F32).to_vector<float>() == std::vector<float>{1, 2, -0.5f, 65504});
    CHECK(g.load(context, "c").shape() == Shape{3});
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("Gguf loads Q8_0 and Q4_0 blocks as quantized matrices", "[gguf]") {
    GgufWriter w;
    // Q8_0, 2 rows of 32: scales 0.5 (f16 0x3800) and -2 (0xC000).
    std::vector<std::int8_t> q(32);
    for (std::size_t i = 0; i < 32; ++i) q[i] = std::int8_t(int(i) * 8 - 127);
    std::vector<std::uint8_t> q8;
    q8_block(q8, 0x3800, q);
    q8_block(q8, 0xC000, q);
    w.tensor("q8", {32, 2}, kGgmlQ8_0, q8);
    // Q4_0, 1 row of 32: scale 0.25 (0x3400); byte j holds element j in its
    // low nibble and element j + 16 in its high one.
    std::vector<std::uint8_t> q4;
    append(q4, std::uint16_t{0x3400});
    for (std::uint8_t j = 0; j < 16; ++j) q4.push_back(std::uint8_t(j | ((15 - j) << 4)));
    w.tensor("q4", {32, 1}, kGgmlQ4_0, q4);
    const auto path = temp_path("vkml_quant.gguf");
    w.write(path);

    vkml::Context context;
    const Gguf g{path};
    CHECK(g.type_name("q8") == "q8_0");
    const vkml::QuantizedMatrix m8 = g.load_quantized(context, "q8");
    CHECK(m8.type == vkml::QuantType::q8_0);
    CHECK(m8.rows == 2);
    CHECK(m8.cols == 32);
    // The file's f16 scales, as they are.
    CHECK(m8.scales.dtype() == DType::F16);
    CHECK(half_bits(m8.scales) == std::vector<std::uint16_t>{0x3800, 0xC000});
    std::vector<float> want;
    for (const float scale : {0.5f, -2.0f}) {
        for (const std::int8_t v : q) want.push_back(float(v) * scale);
    }
    CHECK(vkml::dequantize(m8).to_vector<float>() == want);

    const vkml::QuantizedMatrix m4 = g.load_quantized(context, "q4");
    CHECK(m4.type == vkml::QuantType::q4_0);
    CHECK(half_bits(m4.scales) == std::vector<std::uint16_t>{0x3400});
    want.clear();
    for (int j = 0; j < 16; ++j) want.push_back(float(j - 8) * 0.25f);
    for (int j = 0; j < 16; ++j) want.push_back(float(15 - j - 8) * 0.25f);
    CHECK(vkml::dequantize(m4).to_vector<float>() == want);

    // Decoded on the host to f16 (as a large embedding table is), the values
    // are the same: all of them fit f16 exactly.
    const vkml::Tensor h4 = g.load_f16(context, "q4");
    CHECK(h4.dtype() == DType::F16);
    CHECK(h4.shape() == vkml::Shape{1, 32});
    CHECK(vkml::cast(h4, DType::F32).to_vector<float>() == want);
    std::vector<float> want8;
    for (const float scale : {0.5f, -2.0f}) {
        for (const std::int8_t v : q) want8.push_back(float(v) * scale);
    }
    CHECK(vkml::cast(g.load_f16(context, "q8"), DType::F32).to_vector<float>() == want8);

    // Matrix multiplication works on them as on vkml's own quantized weights.
    const vkml::Tensor x =
        vkml::Tensor::from_data<float>(context, std::vector<float>(32, 1.0f), {1, 32});
    // Both halves sum to (0 + ... + 15 - 16 * 8) / 4 = -2.
    CHECK(vkml::matmul_transposed(x, m4).to_vector<float>() == std::vector<float>{-4.0f});
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("Gguf rejects what it cannot read", "[gguf]") {
    vkml::Context context;
    GgufWriter w;
    w.tensor("k", {256, 1}, kGgmlQ5_K, std::vector<std::uint8_t>(176, 0));
    w.tensor("q8", {32, 1}, kGgmlQ8_0, std::vector<std::uint8_t>(34, 0));
    w.tensor("f", {2}, kGgmlF32, raw(std::vector<float>{1, 2}));
    const auto path = temp_path("vkml_reject.gguf");
    w.write(path);
    const Gguf g{path};
    CHECK(g.type_name("k") == "q5_K");  // decoded on the host (load_f32), not kept quantized
    REQUIRE_THROWS_WITH(g.load_quantized(context, "k"), ContainsSubstring("q5_K"));
    REQUIRE_THROWS_WITH(g.load(context, "q8"), ContainsSubstring("load_quantized"));
    REQUIRE_THROWS_WITH(g.load_quantized(context, "f"), ContainsSubstring("f32"));
    REQUIRE_THROWS_WITH(g.shape("missing"), ContainsSubstring("missing"));

    const auto bad = temp_path("vkml_bad_magic.gguf");
    std::ofstream(bad, std::ios::binary) << "GGML and then some bytes";
    REQUIRE_THROWS_WITH(Gguf{bad}, ContainsSubstring("not a GGUF file"));

    // Version 1 wrote 32-bit counts; vkml reads 2 and 3.
    std::vector<std::uint8_t> v1;
    append(v1, std::uint32_t{0x46554747});
    append(v1, std::uint32_t{1});
    append(v1, std::uint32_t{0});
    append(v1, std::uint32_t{0});
    const auto old = temp_path("vkml_v1.gguf");
    std::ofstream(old, std::ios::binary)
        .write(reinterpret_cast<const char*>(v1.data()), std::streamsize(v1.size()));
    REQUIRE_THROWS_WITH(Gguf{old}, ContainsSubstring("version 1"));

    // Cut short in the middle of the metadata.
    GgufWriter t;
    t.string("general.architecture", "llama");
    const auto cut = temp_path("vkml_cut.gguf");
    t.write(cut);
    std::filesystem::resize_file(cut, 30);
    REQUIRE_THROWS_WITH(Gguf{cut}, ContainsSubstring("truncated"));
}

TEST_CASE("k-quant blocks decode as gguf-py decodes them", "[gguf]") {
    // Random blocks, and gguf-py's values for them (tools/gen_kquant_cases.py).
    std::ifstream in(std::string(VKML_TEST_DATA_DIR) + "/kquant_cases.json");
    REQUIRE(in);
    const nlohmann::json cases = nlohmann::json::parse(in);
    REQUIRE(cases.size() == 6);
    for (const auto& c : cases) {
        const std::string type = c.at("type");
        CAPTURE(type);
        const std::string hex = c.at("bytes");
        std::vector<std::uint8_t> bytes;
        for (std::size_t i = 0; i < hex.size(); i += 2) {
            bytes.push_back(std::uint8_t(std::stoi(hex.substr(i, 2), nullptr, 16)));
        }
        const auto want = c.at("values").get<std::vector<float>>();
        const std::vector<float> got = vkml::detail::dequantize_ggml(type, bytes);
        REQUIRE(got.size() == want.size());
        std::size_t bad = 0;
        for (std::size_t i = 0; i < got.size(); ++i) {
            if (!(std::abs(got[i] - want[i]) <= 1e-6f * (1.0f + std::abs(want[i])))) ++bad;
        }
        CHECK(bad == 0);
    }
    REQUIRE_THROWS_WITH(vkml::detail::dequantize_ggml("q2_K", std::vector<std::uint8_t>(84)),
                        ContainsSubstring("q2_K"));
}

TEST_CASE("Gguf loads q5 and q6 tensors as Q8_0 matrices of their values", "[gguf]") {
    // Decoded and requantized on the host, with no f32 copy on the GPU: each
    // value within half a Q8_0 step of gguf-py's (tools/gen_kquant_cases.py).
    std::ifstream in(std::string(VKML_TEST_DATA_DIR) + "/kquant_cases.json");
    REQUIRE(in);
    const nlohmann::json cases = nlohmann::json::parse(in);
    const std::map<std::string, GgmlType> types{
        {"q5_0", kGgmlQ5_0}, {"q5_1", kGgmlQ5_1}, {"q5_K", kGgmlQ5_K}, {"q6_K", kGgmlQ6_K}};
    vkml::Context context;
    std::size_t seen = 0;
    for (const auto& c : cases) {
        const std::string type = c.at("type");
        if (!types.contains(type)) continue;
        ++seen;
        CAPTURE(type);
        const std::string hex = c.at("bytes");
        std::vector<std::uint8_t> bytes;
        for (std::size_t i = 0; i < hex.size(); i += 2) {
            bytes.push_back(std::uint8_t(std::stoi(hex.substr(i, 2), nullptr, 16)));
        }
        const auto want = c.at("values").get<std::vector<float>>();
        // Two rows where each holds whole Q8_0 blocks, so a k-quant block spans both.
        const std::size_t rows = want.size() / 2 % 32 == 0 ? 2 : 1;
        const std::size_t cols = want.size() / rows;
        GgufWriter w;
        w.tensor("t", {cols, rows}, types.at(type), bytes);
        const auto path = temp_path("vkml_q8_" + type + ".gguf");
        w.write(path);

        const vkml::QuantizedMatrix q = Gguf{path}.load_q8(context, "t");
        CHECK(q.type == vkml::QuantType::q8_0);
        CHECK(q.scales.dtype() == DType::F16);
        CHECK(q.rows == std::int64_t(rows));
        CHECK(q.cols == std::int64_t(cols));
        const std::vector<float> got = vkml::dequantize(q).to_vector<float>();
        REQUIRE(got.size() == want.size());
        std::size_t bad = 0;
        for (std::size_t b = 0; b < want.size(); b += 32) {
            float absmax = 0;
            for (std::size_t i = b; i < b + 32; ++i) absmax = std::max(absmax, std::abs(want[i]));
            // The step rounded to f16.
            const float step = absmax / 127 * (1 + 0x1p-11f);
            for (std::size_t i = b; i < b + 32; ++i) {
                if (!(std::abs(got[i] - want[i]) <= step * 0.5001f + 1e-7f)) ++bad;
            }
        }
        CHECK(bad == 0);
    }
    CHECK(seen == 4);
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("Gguf loads q4_1 and q4_K tensors as q4_1 matrices", "[gguf]") {
    // q4_K's 32-value sub-blocks are each q * scale - min: q4_1's form. q4_1
    // loads exactly; q4_K's scale and min, products of two scales each, are
    // rounded to f16 (tools/gen_kquant_cases.py).
    std::ifstream in(std::string(VKML_TEST_DATA_DIR) + "/kquant_cases.json");
    REQUIRE(in);
    const nlohmann::json cases = nlohmann::json::parse(in);
    vkml::Context context;
    for (const auto& c : cases) {
        const std::string type = c.at("type");
        if (type != "q4_1" && type != "q4_K") continue;
        CAPTURE(type);
        const std::string hex = c.at("bytes");
        std::vector<std::uint8_t> bytes;
        for (std::size_t i = 0; i < hex.size(); i += 2) {
            bytes.push_back(std::uint8_t(std::stoi(hex.substr(i, 2), nullptr, 16)));
        }
        const auto want = c.at("values").get<std::vector<float>>();
        GgufWriter w;
        w.tensor("t", {want.size(), 1}, type == "q4_1" ? kGgmlQ4_1 : kGgmlQ4_K, bytes);
        const auto path = temp_path("vkml_" + type + ".gguf");
        w.write(path);

        const vkml::QuantizedMatrix q = Gguf{path}.load_quantized(context, "t");
        CHECK(q.type == vkml::QuantType::q4_1);
        CHECK(q.scales.dtype() == DType::F16);
        const std::vector<float> got = vkml::dequantize(q).to_vector<float>();
        REQUIRE(got.size() == want.size());
        // q * scale + offset with each rounded by at most 2^-11 of itself:
        // within 2^-11 of 15 |scale| + |offset| <= 31 max |value| of the
        // sub-block, as q is at most 15 and the offset its smallest value.
        std::size_t bad = 0;
        for (std::size_t b = 0; b < got.size(); b += 32) {
            float largest = 0.0f;
            for (std::size_t i = b; i < b + 32; ++i) largest = std::max(largest, std::abs(want[i]));
            const float tolerance = type == "q4_1" ? 1e-6f * largest : 31.0f * largest * 0x1p-11f;
            for (std::size_t i = b; i < b + 32; ++i) {
                if (!(std::abs(got[i] - want[i]) <= tolerance + 1e-6f)) ++bad;
            }
        }
        CHECK(bad == 0);
    }
    CHECK(context.validation_error_count() == 0);
}
