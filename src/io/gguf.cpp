#include "io/gguf.hpp"

#include <array>
#include <cstring>
#include <fstream>
#include <span>

#include "vkml/error.hpp"

namespace vkml::detail {

namespace {

// ggml tensor types by number, with the block each packs (values, bytes).
struct GgmlType {
    const char* name;
    std::uint64_t block_values;
    std::uint64_t block_bytes;
};

GgmlType ggml_type(std::uint32_t type) {
    static const std::map<std::uint32_t, GgmlType> types{
        {0, {"f32", 1, 4}},       {1, {"f16", 1, 2}},       {2, {"q4_0", 32, 18}},
        {3, {"q4_1", 32, 20}},    {6, {"q5_0", 32, 22}},    {7, {"q5_1", 32, 24}},
        {8, {"q8_0", 32, 34}},    {9, {"q8_1", 32, 36}},    {10, {"q2_K", 256, 84}},
        {11, {"q3_K", 256, 110}}, {12, {"q4_K", 256, 144}}, {13, {"q5_K", 256, 176}},
        {14, {"q6_K", 256, 210}}, {15, {"q8_K", 256, 292}}, {30, {"bf16", 1, 2}}};
    const auto it = types.find(type);
    if (it != types.end()) return it->second;
    return {"unknown", 0, 0};
}

constexpr std::uint32_t kF32 = 0, kF16 = 1, kQ4_0 = 2, kQ8_0 = 8, kBF16 = 30;

// Reads the header from a stream, failing cleanly on a file cut short.
class Reader {
public:
    Reader(std::ifstream& in, std::string where) : in_(in), where_(std::move(where)) {}

    template <class T>
    T get() {
        T v;
        bytes(&v, sizeof v);
        return v;
    }
    std::string string() {
        const auto size = get<std::uint64_t>();
        if (size > kMaxString) fail("a string of " + std::to_string(size) + " bytes");
        std::string s(size, '\0');
        bytes(s.data(), size);
        return s;
    }
    void bytes(void* out, std::uint64_t n) {
        if (!in_.read(static_cast<char*>(out), static_cast<std::streamsize>(n))) {
            throw Error(where_ + ": truncated");
        }
    }
    std::uint64_t position() { return static_cast<std::uint64_t>(in_.tellg()); }
    [[noreturn]] void fail(const std::string& what) {
        throw Error(where_ + ": malformed header: " + what);
    }

    // A metadata value of type t: 0-12 are u8, i8, u16, i16, u32, i32, f32,
    // bool, string, array, u64, i64, f64.
    nlohmann::json value(std::uint32_t t, int depth = 0) {
        switch (t) {
            case 0: return get<std::uint8_t>();
            case 1: return get<std::int8_t>();
            case 2: return get<std::uint16_t>();
            case 3: return get<std::int16_t>();
            case 4: return get<std::uint32_t>();
            case 5: return get<std::int32_t>();
            case 6: return get<float>();
            case 7: return get<std::uint8_t>() != 0;
            case 8: return string();
            case 9: {
                if (depth > 4) fail("arrays nested too deep");
                const auto element = get<std::uint32_t>();
                const auto count = get<std::uint64_t>();
                if (count > kMaxArray) fail("an array of " + std::to_string(count) + " values");
                nlohmann::json out = nlohmann::json::array();
                for (std::uint64_t i = 0; i < count; ++i) out.push_back(value(element, depth + 1));
                return out;
            }
            case 10: return get<std::uint64_t>();
            case 11: return get<std::int64_t>();
            case 12: return get<double>();
            default: fail("a value of unknown type " + std::to_string(t));
        }
    }

private:
    // Guards against allocating whatever a corrupt length says.
    static constexpr std::uint64_t kMaxString = std::uint64_t{1} << 30;
    static constexpr std::uint64_t kMaxArray = std::uint64_t{1} << 28;

    std::ifstream& in_;
    std::string where_;
};

float f16_to_float(std::uint16_t h) {
    const std::uint32_t sign = std::uint32_t(h & 0x8000u) << 16;
    const std::uint32_t exponent = (h >> 10) & 0x1Fu, mantissa = h & 0x3FFu;
    std::uint32_t bits;
    if (exponent == 0) {  // zero or subnormal: mantissa * 2^-24, exact in f32
        float f = float(mantissa) * 5.9604644775390625e-8f;
        std::memcpy(&bits, &f, 4);
        bits |= sign;
    } else if (exponent == 31) {
        bits = sign | 0x7F800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

}  // namespace

Gguf::Gguf(const std::filesystem::path& path) : path_(path) {
    const std::string where = "gguf: " + path.string();
    std::ifstream in(path, std::ios::binary);
    if (!in) throw Error(where + ": cannot open file");
    std::error_code ec;
    file_size_ = std::filesystem::file_size(path, ec);
    if (ec) throw Error(where + ": cannot read file size");

    Reader r{in, where};
    std::array<char, 4> magic{};
    if (file_size_ < 4 || !in.read(magic.data(), 4) || std::string(magic.data(), 4) != "GGUF") {
        throw Error(where + ": not a GGUF file");
    }
    version_ = r.get<std::uint32_t>();
    if (version_ < 2 || version_ > 3) {
        throw Error(where + ": GGUF version " + std::to_string(version_) +
                    " is not supported (2 and 3 are)");
    }
    const auto tensors = r.get<std::uint64_t>();
    const auto kvs = r.get<std::uint64_t>();
    if (tensors > (std::uint64_t{1} << 24) || kvs > (std::uint64_t{1} << 24)) {
        r.fail("counts of " + std::to_string(tensors) + " tensors and " + std::to_string(kvs) +
               " metadata entries");
    }
    for (std::uint64_t i = 0; i < kvs; ++i) {
        std::string key = r.string();
        const auto type = r.get<std::uint32_t>();
        metadata_[key] = r.value(type);
    }

    std::vector<std::pair<std::string, Entry>> infos;
    for (std::uint64_t i = 0; i < tensors; ++i) {
        std::string name = r.string();
        const auto dims = r.get<std::uint32_t>();
        if (dims > 4) r.fail("tensor " + name + " has " + std::to_string(dims) + " dimensions");
        Shape shape(dims);
        for (std::uint32_t d = 0; d < dims; ++d) {
            shape[dims - 1 - d] = static_cast<std::int64_t>(r.get<std::uint64_t>());
        }
        const auto type = r.get<std::uint32_t>();
        const auto offset = r.get<std::uint64_t>();
        infos.emplace_back(std::move(name), Entry{std::move(shape), type, offset});
    }

    const std::uint64_t alignment = metadata_.value("general.alignment", std::uint64_t{32});
    if (alignment == 0) r.fail("general.alignment is 0");
    data_start_ = (r.position() + alignment - 1) / alignment * alignment;
    for (auto& [name, e] : infos) {
        const GgmlType t = ggml_type(e.type);
        std::uint64_t count = 1;
        for (const std::int64_t d : e.shape) count *= static_cast<std::uint64_t>(d);
        // Sizes are known for the types vkml names; the rest are checked
        // only to start inside the file.
        const std::uint64_t bytes =
            t.block_values ? (count + t.block_values - 1) / t.block_values * t.block_bytes : 0;
        if (data_start_ + e.offset + bytes > file_size_) {
            throw Error(where + ": tensor " + name + " runs past the end of the file");
        }
        entries_.emplace(std::move(name), std::move(e));
    }
}

std::vector<std::string> Gguf::names() const {
    std::vector<std::string> out;
    for (const auto& [name, e] : entries_) out.push_back(name);
    return out;
}

bool Gguf::contains(const std::string& name) const { return entries_.contains(name); }

const Gguf::Entry& Gguf::entry(const std::string& name) const {
    const auto it = entries_.find(name);
    if (it == entries_.end()) {
        throw Error("gguf: " + path_.string() + " has no tensor \"" + name + "\"");
    }
    return it->second;
}

const Shape& Gguf::shape(const std::string& name) const { return entry(name).shape; }

std::string Gguf::type_name(const std::string& name) const {
    const Entry& e = entry(name);
    const GgmlType t = ggml_type(e.type);
    return t.block_values ? t.name : "type " + std::to_string(e.type);
}

std::vector<std::uint8_t> Gguf::read(const std::string& name, std::uint64_t bytes) const {
    std::vector<std::uint8_t> out(bytes);
    std::ifstream in(path_, std::ios::binary);
    in.seekg(static_cast<std::streamoff>(data_start_ + entry(name).offset));
    if (!in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes))) {
        throw Error("gguf: reading tensor \"" + name + "\" from " + path_.string() + " failed");
    }
    return out;
}

Tensor Gguf::load(Context& context, const std::string& name) const {
    const Entry& e = entry(name);
    DType dtype;
    if (e.type == kF32) {
        dtype = DType::F32;
    } else if (e.type == kF16) {
        dtype = DType::F16;
    } else if (e.type == kBF16) {
        dtype = DType::BF16;
    } else {
        throw Error("gguf: tensor \"" + name + "\" is " + type_name(name) +
                    "; load_quantized reads q8_0 and q4_0, and other types are not supported");
    }
    std::int64_t count = 1;
    for (const std::int64_t d : e.shape) count *= d;
    const auto bytes = read(name, std::uint64_t(count) * element_size(dtype));
    return Tensor::from_bytes(context, std::as_bytes(std::span{bytes}), e.shape, dtype);
}

QuantizedMatrix Gguf::load_quantized(Context& context, const std::string& name) const {
    const Entry& e = entry(name);
    if (e.type != kQ8_0 && e.type != kQ4_0) {
        throw Error("gguf: tensor \"" + name + "\" is " + type_name(name) +
                    ", which vkml cannot load quantized (q8_0 and q4_0 it can)");
    }
    if (e.shape.size() != 2 || e.shape[1] % 32 != 0) {
        throw Error("gguf: tensor \"" + name + "\" of shape " + to_string(e.shape) +
                    " is not a matrix of whole blocks");
    }
    const std::int64_t rows = e.shape[0], cols = e.shape[1];
    const std::int64_t blocks = rows * cols / 32;
    const bool q8 = e.type == kQ8_0;
    const auto data = read(name, std::uint64_t(blocks) * (q8 ? 34 : 18));

    // Each block: an f16 scale, then its values. Q8_0's int8 values are in
    // order, as vkml keeps them; Q4_0's byte j holds value j (low nibble) and
    // j + 16 (high), where vkml's words hold eight in a row, byte k value k
    // and k + 4 (see QuantizedMatrix).
    std::vector<float> scales(static_cast<std::size_t>(blocks));
    std::vector<std::uint32_t> words(std::size_t(blocks) * (q8 ? 8 : 4));
    for (std::int64_t b = 0; b < blocks; ++b) {
        const std::uint8_t* block = data.data() + std::size_t(b) * (q8 ? 34 : 18);
        std::uint16_t scale;
        std::memcpy(&scale, block, 2);
        scales[std::size_t(b)] = f16_to_float(scale);
        if (q8) {
            std::memcpy(&words[std::size_t(b) * 8], block + 2, 32);
            continue;
        }
        std::array<std::uint8_t, 32> nibble{};
        for (std::size_t j = 0; j < 16; ++j) {
            nibble[j] = block[2 + j] & 0x0F;
            nibble[j + 16] = block[2 + j] >> 4;
        }
        for (std::size_t w = 0; w < 4; ++w) {
            std::uint32_t word = 0;
            for (std::size_t k = 0; k < 4; ++k) {
                const std::uint32_t byte = nibble[8 * w + k] | (nibble[8 * w + k + 4] << 4);
                word |= byte << (8 * k);
            }
            words[std::size_t(b) * 4 + w] = word;
        }
    }
    const std::int64_t per_word = q8 ? 4 : 8;
    return QuantizedMatrix{.values = Tensor::from_bytes(context, std::as_bytes(std::span{words}),
                                                        {rows, cols / per_word}, DType::I32),
                           .scales = Tensor::from_data<float>(context, scales, {rows, cols / 32}),
                           .rows = rows,
                           .cols = cols,
                           .type = q8 ? QuantType::q8_0 : QuantType::q4_0};
}

}  // namespace vkml::detail
