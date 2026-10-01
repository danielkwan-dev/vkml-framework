// Reading 16-bit floats packed two to a 32-bit word, so kernels need no 16-bit
// storage feature. Element i of a packed array is half i % 2 of word i / 2,
// low half first (little-endian). Values widen to f32 exactly.

const uint TYPE_F32 = 0;  // DType values of the element types kernels read
const uint TYPE_F16 = 1;
const uint TYPE_BF16 = 2;
// Q8_0 weights: int8 values packed four to a word, each block of 32 along a
// row sharing one f32 scale, kept in a separate array (see quantize.comp).
const uint TYPE_Q8 = 3;
const uint Q8_BLOCK = 32;
// Q4_0 weights: four-bit values q + 8, eight consecutive ones to a word, in
// blocks of 32 with a scale each like Q8_0. Byte j of a word holds value j in
// its low nibble and value j + 4 in its high one, so masking the word with
// 0x0F0F0F0F, before or after shifting it right by 4, gives values 0..3 or
// 4..7 as bytes in order: ready for four-way int8 dot products.
const uint TYPE_Q4 = 4;
const uint Q4_BLOCK = 32;

// Q4_1 weights: Q4_0's packing of four-bit values q, not offset by 8, and per
// block of 32 a scale d and an offset m, stored as pairs in the scales array:
// each value is q * d + m. llama.cpp's q4_1, and its q4_K, whose 32-value
// sub-blocks have this form.
const uint TYPE_Q4_1 = 5;

// Bit offset of value i % 8 within its Q4_0 word.
uint q4_shift(uint i) { return 8 * (i & 3u) + 4 * ((i >> 2) & 1u); }

// Byte i % 4 of word, sign-extended.
float q8_byte(uint word, uint i) { return float(bitfieldExtract(int(word), int(8 * (i & 3u)), 8)); }

// Value i % 8 of a Q4_0 word, as the signed value it stands for.
float q4_nibble(uint word, uint i) {
    return float(int(bitfieldExtract(word, int(q4_shift(i)), 4)) - 8);
}

// Value i % 8 of a Q4_1 word: its code q, 0 to 15.
float q4_code(uint word, uint i) { return float(bitfieldExtract(word, int(q4_shift(i)), 4)); }

// Values 4 * half .. 4 * half + 3 of one Q4_1 word: q * dm.x + dm.y.
vec4 q4_1_vec4(uint word, uint half_, vec2 dm) {
    const uint h = word >> (4 * half_);
    return vec4(h & 15u, (h >> 8) & 15u, (h >> 16) & 15u, (h >> 24) & 15u) * dm.x + dm.y;
}

// Values 4 * half .. 4 * half + 3 of one Q4_0 word times their block's scale.
vec4 q4_vec4(uint word, uint half_, float scale) {
    const uint h = word >> (4 * half_);
    return (vec4(h & 15u, (h >> 8) & 15u, (h >> 16) & 15u, (h >> 24) & 15u) - 8.0) * scale;
}

// The four values of one word times their block's scale.
vec4 q8_vec4(uint word, float scale) {
    const int w = int(word);
    return vec4(bitfieldExtract(w, 0, 8), bitfieldExtract(w, 8, 8), bitfieldExtract(w, 16, 8),
                bitfieldExtract(w, 24, 8)) * scale;
}

// Decoded by hand rather than with unpackHalf2x16, which may flush f16
// subnormals to zero.
float f16_to_float(uint h) {
    const uint sign = (h & 0x8000u) << 16;
    const uint exponent = (h >> 10) & 0x1Fu;
    const uint mantissa = h & 0x3FFu;
    if (exponent == 0) {  // zero or subnormal: mantissa * 2^-24, exact in f32
        return uintBitsToFloat(floatBitsToUint(float(mantissa) * 5.9604644775390625e-8) | sign);
    }
    if (exponent == 31) return uintBitsToFloat(sign | 0x7F800000u | (mantissa << 13));
    return uintBitsToFloat(sign | ((exponent + 112u) << 23) | (mantissa << 13));
}

// f32 bits to f16 bits, rounding to nearest even.
uint to_f16(uint bits) {
    const uint sign = (bits >> 16) & 0x8000u;
    const uint a = bits & 0x7FFFFFFFu;
    if (a > 0x7F800000u) return sign | 0x7E00u;   // NaN
    if (a >= 0x477FF000u) return sign | 0x7C00u;  // rounds past 65504: infinity
    if (a < 0x38800000u) {
        // Below 2^-14: a subnormal f16, in units of 2^-24.
        const uint e = a >> 23;
        if (e < 102u) return sign;  // under 2^-25: rounds to zero
        const uint m = (a & 0x7FFFFFu) | 0x800000u;  // the value is m * 2^(e - 150)
        const uint shift = 126u - e;                  // 14 to 24
        uint q = m >> shift;
        const uint rest = m & ((1u << shift) - 1u), half_ = 1u << (shift - 1u);
        if (rest > half_ || (rest == half_ && (q & 1u) != 0u)) q += 1u;
        return sign | q;
    }
    // Normal: round the 13 dropped bits, then rebias the exponent (127 to 15).
    return sign | ((a + 0xFFFu + ((a >> 13) & 1u) - 0x38000000u) >> 13);
}

float bf16_to_float(uint h) { return uintBitsToFloat(h << 16); }

float half_to_float(uint h, uint type) {
    return type == TYPE_BF16 ? bf16_to_float(h) : f16_to_float(h);
}

// Element i of a packed 16-bit array given its word.
float half_element(uint word, uint i, uint type) {
    return half_to_float((i & 1u) == 0 ? word & 0xFFFFu : word >> 16, type);
}

// Four consecutive elements, the first at an even index, from two words.
vec4 half_vec4(uvec2 words, uint type) {
    return vec4(half_to_float(words.x & 0xFFFFu, type), half_to_float(words.x >> 16, type),
                half_to_float(words.y & 0xFFFFu, type), half_to_float(words.y >> 16, type));
}
