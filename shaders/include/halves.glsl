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
// Q4_0 weights: four-bit values q - 8 packed eight to a word, low bits first,
// in blocks of 32 with a scale each like Q8_0.
const uint TYPE_Q4 = 4;
const uint Q4_BLOCK = 32;

// Byte i % 4 of word, sign-extended.
float q8_byte(uint word, uint i) { return float(bitfieldExtract(int(word), int(8 * (i & 3u)), 8)); }

// Nibble i % 8 of word, as the signed value it stands for.
float q4_nibble(uint word, uint i) {
    return float(int(bitfieldExtract(word, int(4 * (i & 7u)), 4)) - 8);
}

// Values 4 * half .. 4 * half + 3 of one Q4_0 word times their block's scale.
vec4 q4_vec4(uint word, uint half_, float scale) {
    const uint h = word >> (16 * half_);
    return (vec4(h & 15u, (h >> 4) & 15u, (h >> 8) & 15u, (h >> 12) & 15u) - 8.0) * scale;
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
