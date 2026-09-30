#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <span>
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
    // An f32 tensor's values, read on the host (for small ones).
    std::vector<float> read_f32(const std::string& name) const;
    // Any tensor vkml can decode, as f32: the float types, q8_0 and q4_0, and
    // q4_1, q5_0, q5_1, q4_K, q5_K and q6_K decoded on the host.
    Tensor load_f32(Context& context, const std::string& name) const;
    // The same as f16, with quantized types decoded on the host a few blocks
    // at a time: no f32 copy is made, which a table larger than a GPU buffer
    // can hold as f32 (Gemma 3's 262144 x 1152 embeddings) needs.
    Tensor load_f16(Context& context, const std::string& name) const;

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

// Decodes whole blocks of q4_1, q5_0 or q5_1 (32 values each) or q4_K, q5_K
// or q6_K (256 each), as llama.cpp's dequantize_row functions do.
std::vector<float> dequantize_ggml(const std::string& type, std::span<const std::uint8_t> blocks);

}  // namespace vkml::detail
