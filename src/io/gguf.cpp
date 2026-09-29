#include "io/gguf.hpp"

#include <array>
#include <cstring>
#include <fstream>
#include <optional>
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

std::uint16_t read_u16(const std::uint8_t* p) {
    std::uint16_t v;
    std::memcpy(&v, p, 2);
    return v;
}

// The 6-bit scale and min of sub-block j in q4_K's and q5_K's 12 packed bytes.
void scale_min_k4(int j, const std::uint8_t* q, std::uint8_t& scale, std::uint8_t& min) {
    if (j < 4) {
        scale = q[j] & 63;
        min = q[j + 4] & 63;
    } else {
        scale = std::uint8_t((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        min = std::uint8_t((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

// q6_K: 256 values in 210 bytes: 4 low bits (ql[128]), 2 high bits (qh[64]),
// 16 int8 sub-block scales and an f16 scale, each value (q - 32) * d * scale.
void dequantize_q6_k(const std::uint8_t* block, float* y) {
    const std::uint8_t* ql = block;
    const std::uint8_t* qh = block + 128;
    const auto* sc = reinterpret_cast<const std::int8_t*>(block + 192);
    const float d = f16_to_float(read_u16(block + 208));
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l] = d * float(sc[is]) * float(q1);
            y[l + 32] = d * float(sc[is + 2]) * float(q2);
            y[l + 64] = d * float(sc[is + 4]) * float(q3);
            y[l + 96] = d * float(sc[is + 6]) * float(q4);
        }
        y += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

// q4_K (144 bytes) and q5_K (176): f16 d and dmin, 12 bytes of 6-bit scales
// and mins for 8 sub-blocks of 32, (for q5_K, a fifth bit per value in
// qh[32]), then 4-bit values; each is d * scale * q - dmin * min.
void dequantize_q45_k(const std::uint8_t* block, float* y, bool five) {
    const float d = f16_to_float(read_u16(block));
    const float dmin = f16_to_float(read_u16(block + 2));
    const std::uint8_t* scales = block + 4;
    const std::uint8_t* qh = block + 16;
    const std::uint8_t* q = block + (five ? 48 : 16);
    std::uint8_t u1 = 1, u2 = 2;
    for (int j = 0, is = 0; j < 256; j += 64, is += 2) {
        std::uint8_t sc, m;
        scale_min_k4(is, scales, sc, m);
        const float d1 = d * float(sc), m1 = dmin * float(m);
        scale_min_k4(is + 1, scales, sc, m);
        const float d2 = d * float(sc), m2 = dmin * float(m);
        for (int l = 0; l < 32; ++l) {
            const int high = five && (qh[l] & u1) ? 16 : 0;
            *y++ = d1 * float((q[l] & 0xF) + high) - m1;
        }
        for (int l = 0; l < 32; ++l) {
            const int high = five && (qh[l] & u2) ? 16 : 0;
            *y++ = d2 * float((q[l] >> 4) + high) - m2;
        }
        q += 32;
        u1 = std::uint8_t(u1 << 2);
        u2 = std::uint8_t(u2 << 2);
    }
}

// q4_1 (20 bytes), q5_0 (22) and q5_1 (24): 32 values each, an f16 scale d
// (and, for the _1 types, an f16 offset m), for the q5 types a fifth bit per
// value in 4 bytes, then two 4-bit values per byte: value j in the low
// nibble, j + 16 in the high. Each is q * d + m, or (q - 16) * d for q5_0.
void dequantize_legacy(const std::uint8_t* block, float* y, bool five, bool offset) {
    const float d = f16_to_float(read_u16(block));
    const float m = offset ? f16_to_float(read_u16(block + 2)) : 0.0f;
    const std::uint8_t* after = block + (offset ? 4 : 2);
    std::uint32_t high = 0;
    if (five) std::memcpy(&high, after, 4);
    const std::uint8_t* qs = after + (five ? 4 : 0);
    for (std::uint32_t j = 0; j < 16; ++j) {
        int q0 = qs[j] & 0x0F, q1 = qs[j] >> 4;
        if (five) {
            q0 |= int(((high >> j) << 4) & 0x10);
            q1 |= int((high >> (j + 12)) & 0x10);
        }
        if (offset) {
            y[j] = float(q0) * d + m;
            y[j + 16] = float(q1) * d + m;
        } else {
            y[j] = float(q0 - (five ? 16 : 8)) * d;
            y[j + 16] = float(q1 - (five ? 16 : 8)) * d;
        }
    }
}

// Types decoded on the host: values and bytes per block.
std::optional<std::pair<std::size_t, std::size_t>> host_block(const std::string& type) {
    if (type == "q4_1") return std::pair{32, 20};
    if (type == "q5_0") return std::pair{32, 22};
    if (type == "q5_1") return std::pair{32, 24};
    if (type == "q4_K") return std::pair{256, 144};
    if (type == "q5_K") return std::pair{256, 176};
    if (type == "q6_K") return std::pair{256, 210};
    return std::nullopt;
}

}  // namespace

std::vector<float> dequantize_ggml(const std::string& type, std::span<const std::uint8_t> blocks) {
    const auto layout = host_block(type);
    if (!layout) {
        throw Error("gguf: decoding " + type +
                    " is not implemented (q4_1, q5_0, q5_1, q4_K, q5_K and q6_K are)");
    }
    const auto [values, size] = *layout;
    if (blocks.size() % size != 0) {
        throw Error("gguf: " + std::to_string(blocks.size()) + " bytes are not whole " + type +
                    " blocks");
    }
    std::vector<float> out(blocks.size() / size * values);
    for (std::size_t b = 0; b < blocks.size() / size; ++b) {
        const std::uint8_t* block = blocks.data() + b * size;
        float* y = out.data() + b * values;
        if (type == "q6_K") {
            dequantize_q6_k(block, y);
        } else if (type == "q4_K" || type == "q5_K") {
            dequantize_q45_k(block, y, type == "q5_K");
        } else {
            dequantize_legacy(block, y, type != "q4_1", type != "q5_0");
        }
    }
    return out;
}

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

Tensor Gguf::load_f32(Context& context, const std::string& name) const {
    const Entry& e = entry(name);
    const std::string type = type_name(name);
    if (!host_block(type)) {
        const Tensor t = type == "q8_0" || type == "q4_0"
                             ? dequantize(load_quantized(context, name))
                             : load(context, name);
        return t.dtype() == DType::F32 ? t : cast(t, DType::F32);
    }
    std::int64_t count = 1;
    for (const std::int64_t d : e.shape) count *= d;
    const GgmlType t = ggml_type(e.type);
    if (count % std::int64_t(t.block_values) != 0) {
        throw Error("gguf: tensor \"" + name + "\" is not whole " + type + " blocks");
    }
    const auto bytes = read(name, std::uint64_t(count) / t.block_values * t.block_bytes);
    return Tensor::from_data<float>(context, dequantize_ggml(type, bytes), e.shape);
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
