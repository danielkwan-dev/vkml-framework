#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "vkml/context.hpp"
#include "vkml/tensor.hpp"

namespace vkml {

// Reads tensors from a .safetensors file, the format of HF checkpoints.
//
// Opening parses and checks the header: every tensor's byte range must lie in
// the file and match its dtype and shape. Data is read only by load().
class SafeTensors {
public:
    explicit SafeTensors(const std::filesystem::path& path);

    std::vector<std::string> names() const;  // sorted
    bool contains(const std::string& name) const;
    const Shape& shape(const std::string& name) const;
    const std::string& dtype(const std::string& name) const;  // as written, e.g. "BF16"
    const std::map<std::string, std::string>& metadata() const noexcept { return metadata_; }

    // Uploads one tensor. F32 and I32 load as they are; F16 and BF16 are
    // widened to F32, which is exact, since the ops compute in f32 for now.
    Tensor load(Context& context, const std::string& name) const;

private:
    struct Entry {
        std::string dtype;
        Shape shape;
        std::uint64_t begin;  // byte offsets relative to the start of the data
        std::uint64_t end;
    };

    const Entry& entry(const std::string& name) const;

    std::filesystem::path path_;
    std::uint64_t data_start_ = 0;
    std::map<std::string, Entry> entries_;
    std::map<std::string, std::string> metadata_;
};

}  // namespace vkml
