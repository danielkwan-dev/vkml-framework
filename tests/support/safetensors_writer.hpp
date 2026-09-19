#pragma once

// Writes .safetensors files for tests.

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <vkml/tensor.hpp>

namespace vkml_test {

struct Entry {
    std::string name;
    std::string dtype;  // as the format spells it: "F32", "BF16", ...
    vkml::Shape shape;
    std::vector<std::uint8_t> bytes;
};

template <class T>
std::vector<std::uint8_t> raw(const std::vector<T>& values) {
    std::vector<std::uint8_t> out(values.size() * sizeof(T));
    std::memcpy(out.data(), values.data(), out.size());
    return out;
}

inline std::filesystem::path temp_path(const std::string& name) {
    return std::filesystem::temp_directory_path() / name;
}

// u64 little-endian header size, JSON header, then the data of each entry in order.
inline void write_safetensors(const std::filesystem::path& path, const std::vector<Entry>& entries,
                              const std::string& metadata = "") {
    std::string header = "{";
    std::vector<std::uint8_t> data;
    for (const Entry& e : entries) {
        std::string shape;
        for (std::size_t i = 0; i < e.shape.size(); ++i) {
            shape += (i ? "," : "") + std::to_string(e.shape[i]);
        }
        header += "\"" + e.name + "\":{\"dtype\":\"" + e.dtype + "\",\"shape\":[" + shape +
                  "],\"data_offsets\":[" + std::to_string(data.size()) + "," +
                  std::to_string(data.size() + e.bytes.size()) + "]},";
        data.insert(data.end(), e.bytes.begin(), e.bytes.end());
    }
    header += metadata.empty() ? "" : "\"__metadata__\":" + metadata + ",";
    header.back() = '}';
    if (entries.empty() && metadata.empty()) header = "{}";

    std::ofstream out(path, std::ios::binary);
    const std::uint64_t size = header.size();
    out.write(reinterpret_cast<const char*>(&size), 8);  // little-endian hosts only
    out << header;
    out.write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size()));
}

}  // namespace vkml_test
