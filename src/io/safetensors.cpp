#include "vkml/safetensors.hpp"

#include <array>
#include <cstring>
#include <fstream>
#include <optional>

#include <nlohmann/json.hpp>

#include "io/host_quant.hpp"

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
    static const std::map<std::string, DType> dtypes{
        {"F32", DType::F32}, {"F16", DType::F16}, {"BF16", DType::BF16}, {"I32", DType::I32}};
    const auto dtype = dtypes.find(e.dtype);
    if (dtype == dtypes.end()) {
        throw Error("safetensors: tensor \"" + name + "\" is " + e.dtype +
                    ", which vkml cannot load yet (F32, F16, BF16 and I32 can)");
    }

    return Tensor::from_bytes(context, read(e, name), e.shape, dtype->second);
}

std::vector<std::byte> SafeTensors::read(const Entry& e, const std::string& name) const {
    std::vector<std::byte> bytes(e.end - e.begin);
    std::ifstream in(path_, std::ios::binary);
    in.seekg(static_cast<std::streamoff>(data_start_ + e.begin));
    if (!in.read(reinterpret_cast<char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()))) {
        throw Error("safetensors: reading tensor \"" + name + "\" from " + path_.string() +
                    " failed");
    }
    return bytes;
}

QuantizedMatrix SafeTensors::load_q8(Context& context, const std::string& name) const {
    const Entry& e = entry(name);
    if (e.dtype != "F32" && e.dtype != "F16" && e.dtype != "BF16") {
        throw Error("safetensors: tensor \"" + name + "\" is " + e.dtype +
                    ", which load_q8 cannot quantize (F32, F16 and BF16 it can)");
    }
    if (e.shape.size() != 2 || e.shape[1] % 32 != 0) {
        throw Error("safetensors: tensor \"" + name + "\" of shape " + to_string(e.shape) +
                    " is not a matrix of whole 32-value blocks");
    }
    const std::vector<std::byte> bytes = read(e, name);
    const std::string& dtype = e.dtype;
    return detail::quantize_q8_host(context, e.shape[0], e.shape[1], 32,
                                    [&](std::size_t u, float* y) {
                                        for (std::size_t i = 0; i < 32; ++i) {
                                            const std::size_t n = u * 32 + i;
                                            if (dtype == "F32") {
                                                std::memcpy(&y[i], bytes.data() + 4 * n, 4);
                                                continue;
                                            }
                                            std::uint16_t h;
                                            std::memcpy(&h, bytes.data() + 2 * n, 2);
                                            if (dtype == "F16") {
                                                y[i] = detail::f16_to_float(h);
                                            } else {
                                                const std::uint32_t bits = std::uint32_t(h) << 16;
                                                std::memcpy(&y[i], &bits, 4);
                                            }
                                        }
                                    });
}

}  // namespace vkml
