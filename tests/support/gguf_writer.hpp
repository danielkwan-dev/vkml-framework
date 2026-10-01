#pragma once

// Writes GGUF files (version 3, as llama.cpp does) for tests.

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace vkml_test {

// GGUF metadata value types and the ggml tensor types tests use.
enum GgufType : std::uint32_t {
    kU8 = 0,
    kI8 = 1,
    kU16 = 2,
    kI16 = 3,
    kU32 = 4,
    kI32 = 5,
    kF32 = 6,
    kBool = 7,
    kString = 8,
    kArray = 9,
    kU64 = 10,
    kI64 = 11,
    kF64 = 12
};
enum GgmlType : std::uint32_t {
    kGgmlF32 = 0,
    kGgmlF16 = 1,
    kGgmlQ4_0 = 2,
    kGgmlQ4_1 = 3,
    kGgmlQ5_0 = 6,
    kGgmlQ5_1 = 7,
    kGgmlQ8_0 = 8,
    kGgmlQ2_K = 10,
    kGgmlQ3_K = 11,
    kGgmlQ4_K = 12,
    kGgmlQ5_K = 13,
    kGgmlQ6_K = 14,
    kGgmlBF16 = 30
};

class GgufWriter {
public:
    void u32(const std::string& key, std::uint32_t v) { kv(key, kU32, &v, 4); }
    void i32(const std::string& key, std::int32_t v) { kv(key, kI32, &v, 4); }
    void u64(const std::string& key, std::uint64_t v) { kv(key, kU64, &v, 8); }
    void f32(const std::string& key, float v) { kv(key, kF32, &v, 4); }
    void boolean(const std::string& key, bool v) {
        const std::uint8_t b = v ? 1 : 0;
        kv(key, kBool, &b, 1);
    }
    void string(const std::string& key, const std::string& v) {
        key_and_type(key, kString);
        put_string(meta_, v);
        ++kv_count_;
    }
    void strings(const std::string& key, const std::vector<std::string>& v) {
        key_and_type(key, kArray);
        put(meta_, std::uint32_t{kString});
        put(meta_, std::uint64_t(v.size()));
        for (const std::string& s : v) put_string(meta_, s);
        ++kv_count_;
    }
    void i32s(const std::string& key, const std::vector<std::int32_t>& v) {
        key_and_type(key, kArray);
        put(meta_, std::uint32_t{kI32});
        put(meta_, std::uint64_t(v.size()));
        for (const std::int32_t x : v) put(meta_, x);
        ++kv_count_;
    }
    void f32s(const std::string& key, const std::vector<float>& v) {
        key_and_type(key, kArray);
        put(meta_, std::uint32_t{kF32});
        put(meta_, std::uint64_t(v.size()));
        for (const float x : v) put(meta_, x);
        ++kv_count_;
    }

    // dims as GGUF orders them: the fastest-varying (row length) first.
    void tensor(const std::string& name, const std::vector<std::uint64_t>& dims, GgmlType type,
                const std::vector<std::uint8_t>& bytes) {
        tensors_.push_back({name, dims, type, bytes});
    }

    void write(const std::filesystem::path& path, std::uint64_t alignment = 32) const {
        std::vector<std::uint8_t> out;
        put(out, std::uint32_t{0x46554747});  // "GGUF"
        put(out, std::uint32_t{3});
        put(out, std::uint64_t(tensors_.size()));
        put(out, kv_count_);
        out.insert(out.end(), meta_.begin(), meta_.end());
        std::uint64_t offset = 0;
        std::vector<std::uint64_t> offsets;
        for (const auto& t : tensors_) {
            put_string(out, t.name);
            put(out, std::uint32_t(t.dims.size()));
            for (const std::uint64_t d : t.dims) put(out, d);
            put(out, std::uint32_t{t.type});
            put(out, offset);
            offsets.push_back(offset);
            offset = (offset + t.bytes.size() + alignment - 1) / alignment * alignment;
        }
        out.resize((out.size() + alignment - 1) / alignment * alignment);
        const std::size_t data = out.size();
        for (std::size_t i = 0; i < tensors_.size(); ++i) {
            out.resize(data + offsets[i]);
            out.insert(out.end(), tensors_[i].bytes.begin(), tensors_[i].bytes.end());
        }
        std::ofstream(path, std::ios::binary)
            .write(reinterpret_cast<const char*>(out.data()), std::streamsize(out.size()));
    }

private:
    struct Tensor {
        std::string name;
        std::vector<std::uint64_t> dims;
        GgmlType type;
        std::vector<std::uint8_t> bytes;
    };

    template <class T>
    static void put(std::vector<std::uint8_t>& out, T v) {  // little-endian hosts only
        const auto* p = reinterpret_cast<const std::uint8_t*>(&v);
        out.insert(out.end(), p, p + sizeof(T));
    }
    static void put_string(std::vector<std::uint8_t>& out, const std::string& s) {
        put(out, std::uint64_t(s.size()));
        out.insert(out.end(), s.begin(), s.end());
    }
    void key_and_type(const std::string& key, GgufType type) {
        put_string(meta_, key);
        put(meta_, std::uint32_t{type});
    }
    void kv(const std::string& key, GgufType type, const void* value, std::size_t size) {
        key_and_type(key, type);
        const auto* p = static_cast<const std::uint8_t*>(value);
        meta_.insert(meta_.end(), p, p + size);
        ++kv_count_;
    }

    std::vector<std::uint8_t> meta_;
    std::uint64_t kv_count_ = 0;
    std::vector<Tensor> tensors_;
};

}  // namespace vkml_test
