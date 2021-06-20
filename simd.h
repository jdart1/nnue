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
    for(unsigned i = 0; i < outputSize; i++) {
        __m256i prod = _mm256_setzero_si256();
        const __m256i *w = reinterpret_cast<const __m256i *>(weights[i]);
        for(unsigned j = 0; j < inputSize; j += 32)
        {
            const __m256i *inp = reinterpret_cast<const __m256i *>(&input[j]);
#ifdef VNNI
            prod = _mm256_dpbusd_epi32(prod,inp[0],w[j/32]);
#else
            __m256i x = _mm256_maddubs_epi16(inp[0], w[j/32]);
            x = _mm256_madd_epi16(x, _mm256_set1_epi16(1));
            prod = _mm256_add_epi32(prod, x);
#endif
        }
        __m128i sum = _mm_add_epi32(
                                    _mm256_castsi256_si128(prod), _mm256_extracti128_si256(prod, 1));
        sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x1b));
        output[i] += _mm_cvtsi128_si32(sum) + _mm_extract_epi32(sum, 1);
    }
#else
#error SIMD support requires AVX2
#endif
}

template <size_t size, typename InType, typename OutType>
inline void vec_add(const InType *in,OutType *out) {
    const __m256i *inp = reinterpret_cast<const __m256i *>(in);
    __m256i *outp = reinterpret_cast<__m256i *>(out);
#ifdef AVX2
    for (size_t i = 0; i < (size*8*sizeof(OutType))/simdWidth; ++i) {
        outp[i] = _mm256_add_epi16(outp[i], inp[i]);
    }
#else
#error SIMD support requires AVX2
#endif
}


template <size_t size, typename InType, typename OutType>
inline void vec_sub(const InType *in,OutType *out) {
    const __m256i *inp = reinterpret_cast<const __m256i *>(in);
    __m256i *outp = reinterpret_cast<__m256i *>(out);
#ifdef AVX2
    for (size_t i = 0; i < (size*8*sizeof(OutType))/simdWidth; ++i) {
        outp[i] = _mm256_sub_epi16(outp[i], inp[i]);
    }
#else
#error SIMD support requires AVX2
#endif
}

} // end namespace

#endif
