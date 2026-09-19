#include "vkml/safetensors.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <optional>

#include <nlohmann/json.hpp>

namespace vkml {

namespace {

// Largest header accepted, as in the reference implementation: a guard
// against allocating whatever a corrupt size prefix says.
constexpr std::uint64_t kMaxHeaderBytes = 100'000'000;

// Element sizes of the dtypes the format defines.
std::optional<std::uint64_t> dtype_size(const std::string& dtype) {
    static const std::map<std::string, std::uint64_t> sizes{
        {"BOOL", 1}, {"U8", 1},  {"I8", 1},  {"F8_E4M3", 1}, {"F8_E5M2", 1},
        {"U16", 2},  {"I16", 2}, {"F16", 2}, {"BF16", 2},    {"U32", 4},
        {"I32", 4},  {"F32", 4}, {"U64", 8}, {"I64", 8},     {"F64", 8}};
    const auto it = sizes.find(dtype);
    return it == sizes.end() ? std::nullopt : std::optional{it->second};
}

float half_to_float(std::uint16_t h) {
    const std::uint32_t sign = std::uint32_t(h & 0x8000u) << 16;
    const std::uint32_t exponent = (h >> 10) & 0x1Fu;
    const std::uint32_t mantissa = h & 0x3FFu;
    if (exponent == 0) {  // zero or subnormal: mantissa * 2^-24, exact in f32
        const float magnitude = std::ldexp(static_cast<float>(mantissa), -24);
        return sign ? -magnitude : magnitude;
    }
    const std::uint32_t bits = exponent == 0x1F
                                   ? sign | 0x7F800000u | (mantissa << 13)  // inf or NaN
                                   : sign | ((exponent + 112) << 23) | (mantissa << 13);
    return std::bit_cast<float>(bits);
}

float bfloat16_to_float(std::uint16_t b) { return std::bit_cast<float>(std::uint32_t(b) << 16); }

}  // namespace

SafeTensors::SafeTensors(const std::filesystem::path& path) : path_(path) {
    const std::string where = "safetensors: " + path.string();
    std::ifstream in(path, std::ios::binary);
    if (!in) throw Error(where + ": cannot open file");
    std::error_code ec;
    const std::uint64_t file_size = std::filesystem::file_size(path, ec);
    if (ec) throw Error(where + ": cannot read file size");

    std::array<unsigned char, 8> prefix{};
    if (file_size < 8 || !in.read(reinterpret_cast<char*>(prefix.data()), 8)) {
        throw Error(where + ": too short for the 8-byte header size");
    }
    std::uint64_t header_size = 0;
    for (int i = 7; i >= 0; --i) header_size = header_size << 8 | prefix[std::size_t(i)];
    if (header_size > kMaxHeaderBytes || header_size > file_size - 8) {
        throw Error(where + ": header size " + std::to_string(header_size) +
                    " does not fit in the file");
    }
    data_start_ = 8 + header_size;
    const std::uint64_t data_size = file_size - data_start_;

    std::string header(header_size, '\0');
    in.read(header.data(), static_cast<std::streamsize>(header_size));
    nlohmann::json json;
    try {
        json = nlohmann::json::parse(header);
    } catch (const nlohmann::json::exception& e) {
        throw Error(where + ": header is not valid JSON: " + e.what());
    }
    if (!json.is_object()) throw Error(where + ": header is not a JSON object");

    for (const auto& [name, value] : json.items()) {
        if (name == "__metadata__") {
            for (const auto& [key, text] : value.items()) {
                if (text.is_string()) metadata_[key] = text.get<std::string>();
            }
            continue;
        }
        const std::string what = where + ": tensor \"" + name + "\"";
        try {
            Entry e;
            e.dtype = value.at("dtype").get<std::string>();
            std::uint64_t count = 1;
            for (const auto& dim : value.at("shape")) {
                const auto d = dim.get<std::int64_t>();
                if (d < 0) throw Error(what + " has a negative dimension");
                e.shape.push_back(d);
                count *= static_cast<std::uint64_t>(d);
            }
            const auto& offsets = value.at("data_offsets");
            e.begin = offsets.at(0).get<std::uint64_t>();
            e.end = offsets.at(1).get<std::uint64_t>();
            if (offsets.size() != 2 || e.begin > e.end || e.end > data_size) {
                throw Error(what + " has data offsets outside the file");
            }
            const auto element = dtype_size(e.dtype);
            if (element && e.end - e.begin != count * *element) {
                throw Error(what + " has " + std::to_string(e.end - e.begin) +
                            " bytes, but its shape and dtype need " +
                            std::to_string(count * *element));
            }
            entries_.emplace(name, std::move(e));
        } catch (const nlohmann::json::exception& e) {
            throw Error(what + " has a malformed entry: " + e.what());
        }
    }
}

std::vector<std::string> SafeTensors::names() const {
    std::vector<std::string> out;
    for (const auto& [name, entry] : entries_) out.push_back(name);
    return out;
}

bool SafeTensors::contains(const std::string& name) const { return entries_.contains(name); }

const Shape& SafeTensors::shape(const std::string& name) const { return entry(name).shape; }

const std::string& SafeTensors::dtype(const std::string& name) const { return entry(name).dtype; }

const SafeTensors::Entry& SafeTensors::entry(const std::string& name) const {
    const auto it = entries_.find(name);
    if (it == entries_.end()) {
        throw Error("safetensors: " + path_.string() + " has no tensor \"" + name + "\"");
    }
    return it->second;
}

Tensor SafeTensors::load(Context& context, const std::string& name) const {
    const Entry& e = entry(name);
    const bool widen = e.dtype == "F16" || e.dtype == "BF16";
    if (!widen && e.dtype != "F32" && e.dtype != "I32") {
        throw Error("safetensors: tensor \"" + name + "\" is " + e.dtype +
                    ", which vkml cannot load yet (F32, F16, BF16 and I32 can)");
    }

    std::vector<std::byte> bytes(e.end - e.begin);
    std::ifstream in(path_, std::ios::binary);
    in.seekg(static_cast<std::streamoff>(data_start_ + e.begin));
    if (!in.read(reinterpret_cast<char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()))) {
        throw Error("safetensors: reading tensor \"" + name + "\" from " + path_.string() +
                    " failed");
    }
    if (!widen) {
        return Tensor::from_bytes(context, bytes, e.shape,
                                  e.dtype == "F32" ? DType::F32 : DType::I32);
    }

    std::vector<float> values(bytes.size() / 2);
    for (std::size_t i = 0; i < values.size(); ++i) {
        std::uint16_t raw;
        std::memcpy(&raw, bytes.data() + 2 * i,
                    2);  // the format is little-endian, as are our hosts
        values[i] = e.dtype == "F16" ? half_to_float(raw) : bfloat16_to_float(raw);
    }
    return Tensor::from_data<float>(context, values, e.shape);
}

}  // namespace vkml
