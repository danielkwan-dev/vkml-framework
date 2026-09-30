#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <vkml/vkml.hpp>

#include "support/safetensors_writer.hpp"

using Catch::Matchers::ContainsSubstring;
using vkml::DType;
using vkml::SafeTensors;
using vkml::Shape;
using vkml_test::Entry;
using vkml_test::raw;

namespace {

// Writes a safetensors file under the temp directory and returns its path.
std::filesystem::path write_file(const std::string& file, const std::vector<Entry>& entries,
                                 const std::string& metadata = "") {
    const auto path = vkml_test::temp_path(file);
    vkml_test::write_safetensors(path, entries, metadata);
    return path;
}

std::filesystem::path write_raw(const std::string& file, const std::string& contents) {
    const auto path = std::filesystem::temp_directory_path() / file;
    std::ofstream(path, std::ios::binary) << contents;
    return path;
}

std::string with_header_size(const std::string& header) {
    std::string out(8, '\0');
    const std::uint64_t size = header.size();
    std::memcpy(out.data(), &size, 8);
    return out + header;
}

}  // namespace

TEST_CASE("SafeTensors lists tensors and loads f32 and i32 data unchanged", "[safetensors]") {
    const auto path = write_file(
        "vkml_basic.safetensors",
        {{"layers.0.weight", "F32", {2, 3}, raw(std::vector<float>{1, -2, 3.5f, 4, 5, 6})},
         {"ids", "I32", {3}, raw(std::vector<std::int32_t>{-7, 0, 7})}},
        R"({"format":"pt"})");
    const SafeTensors file{path};

    CHECK(file.names() == std::vector<std::string>{"ids", "layers.0.weight"});
    CHECK(file.contains("ids"));
    CHECK_FALSE(file.contains("nope"));
    CHECK(file.shape("layers.0.weight") == Shape{2, 3});
    CHECK(file.metadata().at("format") == "pt");

    vkml::Context context;
    const vkml::Tensor w = file.load(context, "layers.0.weight");
    CHECK(w.dtype() == DType::F32);
    CHECK(w.shape() == Shape{2, 3});
    CHECK(w.to_vector<float>() == std::vector<float>{1, -2, 3.5f, 4, 5, 6});
    CHECK(file.load(context, "ids").to_vector<std::int32_t>() ==
          std::vector<std::int32_t>{-7, 0, 7});
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("SafeTensors keeps f16 and bf16 data as it is", "[safetensors]") {
    const std::vector<std::uint16_t> f16{0x3C00, 0xC000, 0x7BFF};
    const std::vector<std::uint16_t> bf16{0x3F80, 0xC049, 0x7E96, 0xFF80};
    const auto path = write_file("vkml_half.safetensors",
                                 {{"h", "F16", {3}, raw(f16)}, {"b", "BF16", {2, 2}, raw(bf16)}});
    const SafeTensors file{path};
    vkml::Context context;

    const vkml::Tensor h = file.load(context, "h");
    CHECK(h.dtype() == DType::F16);
    const auto f16_bytes = std::as_bytes(std::span{f16});
    CHECK(h.to_bytes() == std::vector<std::byte>(f16_bytes.begin(), f16_bytes.end()));
    const vkml::Tensor b = file.load(context, "b");
    CHECK(b.dtype() == DType::BF16);
    CHECK(b.shape() == Shape{2, 2});
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("SafeTensors quantizes a float matrix to Q8_0 on the host", "[safetensors]") {
    // As a table too large for a GPU buffer is loaded: each value within half
    // a Q8_0 step of the stored one, whatever the stored type.
    // Multiples of 1/8 below 32, at most 8 significant bits: exact in f16 and
    // bf16 alike.
    std::vector<float> values(3 * 64);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = float(int((i * 37) % 97) * (i % 5 == 0 ? 2 : 1) - 48) / 8.0f;
    }
    std::vector<std::uint16_t> bf16(values.size()), f16(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        std::uint32_t bits;
        std::memcpy(&bits, &values[i], 4);
        bf16[i] = std::uint16_t(bits >> 16);
        const std::uint32_t exponent = (bits >> 23) & 0xFFu;
        f16[i] = values[i] == 0.0f
                     ? 0
                     : std::uint16_t(((bits >> 16) & 0x8000u) | ((exponent - 112u) << 10) |
                                     ((bits & 0x7FFFFFu) >> 13));
    }
    vkml::Context context;
    const auto path = write_file("vkml_st_q8.safetensors",
                                 {{"f32", "F32", {3, 64}, raw(values)},
                                  {"bf16", "BF16", {3, 64}, raw(bf16)},
                                  {"f16", "F16", {3, 64}, raw(f16)},
                                  {"narrow", "F32", {2, 16}, raw(std::vector<float>(32))},
                                  {"ids", "I32", {1, 32}, raw(std::vector<std::int32_t>(32))}});
    const SafeTensors st{path};
    for (const char* name : {"f32", "bf16", "f16"}) {
        CAPTURE(name);
        const std::vector<float>& want = values;
        CHECK(vkml::cast(st.load(context, name), DType::F32).to_vector<float>() == want);
        const vkml::QuantizedMatrix q = st.load_q8(context, name);
        CHECK(q.type == vkml::QuantType::q8_0);
        CHECK(q.rows == 3);
        CHECK(q.cols == 64);
        const std::vector<float> got = vkml::dequantize(q).to_vector<float>();
        REQUIRE(got.size() == want.size());
        std::size_t bad = 0;
        for (std::size_t b = 0; b < want.size(); b += 32) {
            float absmax = 0;
            for (std::size_t i = b; i < b + 32; ++i) absmax = std::max(absmax, std::abs(want[i]));
            for (std::size_t i = b; i < b + 32; ++i) {
                if (!(std::abs(got[i] - want[i]) <= absmax / 127 * 0.5001f + 1e-7f)) ++bad;
            }
        }
        CHECK(bad == 0);
    }
    REQUIRE_THROWS_WITH(st.load_q8(context, "narrow"), ContainsSubstring("[2, 16]"));
    REQUIRE_THROWS_WITH(st.load_q8(context, "ids"), ContainsSubstring("I32"));
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("SafeTensors reports missing tensors and unsupported dtypes", "[safetensors]") {
    const auto path = write_file(
        "vkml_errors.safetensors",
        {{"x", "F64", {1}, raw(std::vector<double>{1.0})}, {"empty", "F32", {0, 4}, {}}});
    const SafeTensors file{path};
    vkml::Context context;
    REQUIRE_THROWS_WITH(file.load(context, "missing"), ContainsSubstring("missing"));
    REQUIRE_THROWS_WITH(file.load(context, "x"), ContainsSubstring("F64"));
    CHECK(file.load(context, "empty").shape() == Shape{0, 4});
}

TEST_CASE("SafeTensors rejects malformed files without reading past them", "[safetensors]") {
    SECTION("missing file") {
        REQUIRE_THROWS_WITH(SafeTensors{std::filesystem::temp_directory_path() / "vkml_nope.st"},
                            ContainsSubstring("vkml_nope.st"));
    }
    SECTION("shorter than the size prefix") {
        REQUIRE_THROWS_WITH(SafeTensors{write_raw("vkml_short.safetensors", "abc")},
                            ContainsSubstring("header"));
    }
    SECTION("header size larger than the file") {
        REQUIRE_THROWS_WITH(
            SafeTensors{write_raw("vkml_big.safetensors", with_header_size("{}").substr(0, 8))},
            ContainsSubstring("header"));
    }
    SECTION("header that is not JSON") {
        REQUIRE_THROWS_WITH(
            SafeTensors{write_raw("vkml_json.safetensors", with_header_size("{oops"))},
            ContainsSubstring("JSON"));
    }
    SECTION("data offsets outside the file") {
        const std::string header = R"({"w":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}})";
        REQUIRE_THROWS_WITH(
            SafeTensors{write_raw("vkml_trunc.safetensors", with_header_size(header) + "12345678")},
            ContainsSubstring("\"w\""));
    }
    SECTION("byte range that does not match the shape") {
        const std::string header = R"({"w":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}})";
        REQUIRE_THROWS_WITH(
            SafeTensors{write_raw("vkml_size.safetensors", with_header_size(header) + "12345678")},
            ContainsSubstring("\"w\""));
    }
}
