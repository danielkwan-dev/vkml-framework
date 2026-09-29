#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vkml/context.hpp"
#include "vkml/ops.hpp"
#include "vkml/tensor.hpp"

namespace vkml::detail {

// A GGUF file, llama.cpp's format (versions 2 and 3): typed metadata, then
// tensors in ggml's types. Reads the header on construction; tensors load on
// request, float ones as they are and Q8_0 and Q4_0 ones repacked into
// QuantizedMatrix's layout.
class Gguf {
public:
    explicit Gguf(const std::filesystem::path& path);

    std::uint32_t version() const noexcept { return version_; }

    // The key-value metadata: numbers, bools, strings and arrays of them.
    const nlohmann::json& metadata() const noexcept { return metadata_; }

    std::vector<std::string> names() const;  // sorted
    bool contains(const std::string& name) const;
    // Row-major, as vkml's tensors: GGUF lists dimensions fastest first.
    const Shape& shape(const std::string& name) const;
    std::string type_name(const std::string& name) const;  // ggml's, e.g. "q8_0"

    // An f32, f16 or bf16 tensor, as stored.
    Tensor load(Context& context, const std::string& name) const;
    // A Q8_0 or Q4_0 matrix, repacked: scales widen from f16 to f32 exactly.
    QuantizedMatrix load_quantized(Context& context, const std::string& name) const;

private:
    struct Entry {
        Shape shape;
        std::uint32_t type;
        std::uint64_t offset;  // from the start of the data
    };

    const Entry& entry(const std::string& name) const;
    std::vector<std::uint8_t> read(const std::string& name, std::uint64_t bytes) const;

    std::filesystem::path path_;
    std::uint32_t version_ = 0;
    nlohmann::json metadata_ = nlohmann::json::object();
    std::map<std::string, Entry> entries_;
    std::uint64_t data_start_ = 0;
    std::uint64_t file_size_ = 0;
};

}  // namespace vkml::detail
