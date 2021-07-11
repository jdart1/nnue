// Copyright 2021 by Jon Dart. All Rights Reserved.
#ifndef NNUE_SIMD_H
#define NNUE_SIMD_H

extern "C" {
#include <immintrin.h>
}

namespace simd {
static constexpr size_t simdWidth = 256;

static const __m256i ones256 = _mm256_set1_epi16(1);

inline void dotProduct32x1(const uint8_t *input, const int8_t *weights,
                           const int32_t *biases, int32_t *output) {
    const __m256i *iv = reinterpret_cast<const __m256i *>(input);
    const __m256i *row = reinterpret_cast<const __m256i *>(weights);
#ifdef AVX2
#ifdef VNNI
    __m256i prod = _mm256_dpbusd_epi32(_mm256_setzero_si256(), iv[0], row[0]);
#else
    __m256i prod = _mm256_maddubs_epi16(iv[0], row[0]);
    prod = _mm256_madd_epi16(prod, _mm256_set1_epi16(1));
#endif
    __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(prod),
                                   _mm256_extracti128_si256(prod, 1));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_PERM_BADC));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_PERM_CDAB));
    *output = _mm_cvtsi128_si32(sum128) + biases[0];
#else
#error SIMD support requires AVX2
#endif
}

template <size_t inputSize, size_t outputSize>
inline void dotProductnx32(const uint8_t *input,
                           const int8_t weights[outputSize][inputSize],
                           const int32_t *biases, int32_t *output) {
#ifdef AVX2
    assert(outputSize % 32 == 0 && inputSize % 32 == 0);
    std::memcpy(output, biases, outputSize * 4);
    for (unsigned i = 0; i < outputSize; i++) {
        __m256i prod = _mm256_setzero_si256();
        const __m256i *w = reinterpret_cast<const __m256i *>(weights[i]);
        for (unsigned j = 0; j < inputSize; j += 32) {
            const __m256i *inp = reinterpret_cast<const __m256i *>(&input[j]);
#ifdef VNNI
            prod = _mm256_dpbusd_epi32(prod, inp[0], w[j / 32]);
#else
            __m256i x = _mm256_maddubs_epi16(inp[0], w[j / 32]);
            x = _mm256_madd_epi16(x, _mm256_set1_epi16(1));
            prod = _mm256_add_epi32(prod, x);
#endif
        }
        __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(prod),
                                    _mm256_extracti128_si256(prod, 1));
        sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x1b));
        output[i] += _mm_cvtsi128_si32(sum) + _mm_extract_epi32(sum, 1);
    }
#else
#error SIMD support requires AVX2
#endif
}

template <size_t size, typename InType, typename OutType>
inline void vec_add(const InType *in, OutType *out) {
    const __m256i *inp = reinterpret_cast<const __m256i *>(in);
    __m256i *outp = reinterpret_cast<__m256i *>(out);
#ifdef AVX2
    for (size_t i = 0; i < (size * 8 * sizeof(OutType)) / simdWidth; ++i) {
        outp[i] = _mm256_add_epi16(outp[i], inp[i]);
    }
#else
#error SIMD support requires AVX2
#endif
}

template <size_t size, typename InType, typename OutType>
inline void vec_sub(const InType *in, OutType *out) {
    const __m256i *inp = reinterpret_cast<const __m256i *>(in);
    __m256i *outp = reinterpret_cast<__m256i *>(out);
#ifdef AVX2
    for (size_t i = 0; i < (size * 8 * sizeof(OutType)) / simdWidth; ++i) {
        outp[i] = _mm256_sub_epi16(outp[i], inp[i]);
    }
#else
#error SIMD support requires AVX2
#endif
}

template <size_t size, typename InType, typename OutType>
inline void clamp(const InType *in, OutType *out, InType /*clampMax*/) {
#ifdef AVX2
    assert(sizeof(InType)==2);
    assert(sizeof(OutType)==1);
    const __m256i *inp = reinterpret_cast<const __m256i *>(in);
    __m256i *outp = reinterpret_cast<__m256i *>(out);
    constexpr int control = 0b11011000;
    const __m256i zero = _mm256_setzero_si256();
    for (size_t i = 0; i < (size * 8 * sizeof(OutType)) / simdWidth; ++i) {
        // load 2x256 bit registers of input data
        __m256i words0 = _mm256_load_si256(
            reinterpret_cast<const __m256i *>(inp + 2 * i + 0));
        __m256i words1 = _mm256_load_si256(
            reinterpret_cast<const __m256i *>(inp + 2 * i + 1));
        // clamp and store into one 256-bit output chunk
        _mm256_store_si256(
            &outp[i],
            _mm256_permute4x64_epi64(
                _mm256_max_epi8(_mm256_packs_epi16(words0, words1), zero),
                control));
    }
#else
#error SIMD support requires AVX2
#endif
}

template <size_t size, typename InType, typename OutType>
inline void scale_and_clamp(const InType *in, OutType *out, unsigned rshift, InType /*clampMax*/) {
#ifdef AVX2
    assert(sizeof(InType)==4);
    assert(sizeof(OutType)==1);
    __m256i *inp = const_cast<__m256i *>(reinterpret_cast<const __m256i*>(in));
    __m256i *outp = reinterpret_cast<__m256i *>(out);
    const __m256i control = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
    const __m256i zero = _mm256_setzero_si256();
    for (size_t i = 0; i < (size * 8 * sizeof(OutType)) / simdWidth; ++i) {
        // load 2x256 bit registers of shifted input data (32 bit input, 16 bit output)
        __m256i r1  = _mm256_srai_epi16(_mm256_packs_epi32(inp[4*i + 0],inp[4*i + 1]), rshift);
        __m256i r2  = _mm256_srai_epi16(_mm256_packs_epi32(inp[4*i + 2],inp[4*i + 3]), rshift);
        // clamp and store into one 256-bit output chunk
        outp[i] = _mm256_permutevar8x32_epi32(_mm256_max_epi8(_mm256_packs_epi16(r1, r2), zero), control);
    }
#else
#error SIMD support requires AVX2
#endif
}

} // namespace simd

#endif
