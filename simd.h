// Copyright 2021, 2022 by Jon Dart. All Rights Reserved.
#ifndef NNUE_SIMD_H
#define NNUE_SIMD_H

extern "C" {
#include <immintrin.h>
}

namespace simd {

#ifdef AVX512
    using vec_t = __m512i;
    static constexpr size_t simdWidth = 512;
    static const vec_t ones512 = _mm512_set1_epi16(1);
    static const __m256i ones256 = _mm256_set1_epi16(1);
#elif defined(AVX2)
    using vec_t = __m256i;
    static constexpr size_t simdWidth = 256;
    static const vec_t ones256 = _mm256_set1_epi16(1);
#elif defined(SSE2) || defined(SSSE3)
    using vec_t = __m128i;
    static const vec_t ones128 = _mm_set1_epi16(1);
    static constexpr size_t simdWidth = 128;
#else
#error must set at least one of: AVX2, SSSE3 or SSE2
#endif

template <typename T,unsigned simdWidth>
static inline size_t chunks(unsigned len) {
    return (len * 8 * sizeof(T)) / simdWidth;
}

#ifdef AVX512
void inline mm512_add_dpbusd_epi32(__m512i& acc, __m512i a, __m512i b) {
#ifdef AVX512_VNNI
    acc = _mm512_dpbusd_epi32(acc, a, b);
#else
    __m512i x = _mm512_maddubs_epi16(a, b);
    x = _mm512_madd_epi16(x, ones512);
    acc = _mm512_add_epi32(acc, x);
#endif
}
#endif

#ifdef AVX2
[[maybe_unused]] void inline mm256_add_dpbusd_epi32(__m256i& acc, __m256i a, __m256i b) {
#ifdef VNNI
    acc = _mm256_dpbusd_epi32(acc, a, b);
#else
    __m256i x = _mm256_maddubs_epi16(a, b);
    x = _mm256_madd_epi16(x, ones256);
    acc = _mm256_add_epi32(acc, x);
#endif
}
#endif

static inline void dotProduct32x1(const uint8_t *input, const int8_t *weights,
                                  const int32_t *biases, int32_t *output) {
    // No use using AVX512 here because the input is only 32x8 = 256 bits
#if defined(AVX2)
    const __m256i *inp = reinterpret_cast<const __m256i *>(input);
    const __m256i *row = reinterpret_cast<const __m256i *>(weights);
#ifdef VNNI
    __m256i prod = _mm256_dpbusd_epi32(_mm256_setzero_si256(), inp[0], row[0]);
#else
    __m256i prod = _mm256_maddubs_epi16(inp[0], row[0]);
    prod = _mm256_madd_epi16(prod, ones256);
#endif
    __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(prod),
                                   _mm256_extracti128_si256(prod, 1));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_PERM_BADC));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_PERM_CDAB));
    *output = _mm_cvtsi128_si32(sum128) + biases[0];
#elif defined(SSSE3)
    const vec_t *inp = reinterpret_cast<const vec_t *>(input);
    const vec_t *row = reinterpret_cast<const vec_t *>(weights);
    vec_t p0 = _mm_madd_epi16(_mm_maddubs_epi16(inp[0], row[0]), ones128);
    vec_t p1 = _mm_madd_epi16(_mm_maddubs_epi16(inp[1], row[1]), ones128);
    vec_t sum = _mm_add_epi32(p0, p1);
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0xb));
#ifdef SSE41
    output[0] = _mm_cvtsi128_si32(sum) + _mm_extract_epi32(sum, 1) + biases[0];
#else
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x1));
    output[0] = _mm_cvtsi128_si32(sum) + biases[0];
#endif
#elif defined(SSE2)
    const vec_t zeros = _mm_setzero_si128();
    vec_t sum_lo, sum_hi;
    sum_lo = sum_hi = zeros;
    const auto row = reinterpret_cast<const vec_t*>(weights);
    const vec_t *inp = reinterpret_cast<const vec_t*>(input);
    constexpr unsigned inputSize = 32;
    for (unsigned j = 0; j < chunks<uint8_t,simdWidth>(inputSize); ++j) {
        __m128i row_j = _mm_load_si128(&row[j]);
        __m128i input_j = _mm_load_si128(&inp[j]);
        __m128i row_signs = _mm_cmpgt_epi8(zeros, row_j);
        __m128i extended_row_lo = _mm_unpacklo_epi8(row_j, row_signs);
        __m128i extended_row_hi = _mm_unpackhi_epi8(row_j, row_signs);
        __m128i extended_input_lo = _mm_unpacklo_epi8(input_j, zeros);
        __m128i extended_input_hi = _mm_unpackhi_epi8(input_j, zeros);
        __m128i product_lo = _mm_madd_epi16(extended_row_lo, extended_input_lo);
        __m128i product_hi = _mm_madd_epi16(extended_row_hi, extended_input_hi);
        sum_lo = _mm_add_epi32(sum_lo, product_lo);
        sum_hi = _mm_add_epi32(sum_hi, product_hi);
    }
    __m128i sum = _mm_add_epi32(sum_lo, sum_hi);
    __m128i sum_high_64 = _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 0, 3, 2));
    sum = _mm_add_epi32(sum, sum_high_64);
    __m128i sum_second_32 = _mm_shufflelo_epi16(sum, _MM_SHUFFLE(1, 0, 3, 2));
    sum = _mm_add_epi32(sum, sum_second_32);
    output[0] = _mm_cvtsi128_si32(sum) + biases[0];
#endif
}

template <size_t inputSize, size_t outputSize>
inline void dotProductnx32(const uint8_t *input,
                           const int8_t weights[outputSize][inputSize],
                           const int32_t *biases, int32_t *output) {
#ifdef AVX512
    if constexpr (inputSize >= 64 && inputSize % 64 == 0) {
        std::memcpy(output, biases, outputSize * 4);
        for (unsigned i = 0; i < outputSize; i++) {
  	    vec_t prod = _mm512_setzero_si512();
            const vec_t *w = reinterpret_cast<const vec_t *>(weights[i]);
            for (unsigned j = 0; j < inputSize; j += 64) {
                const vec_t *inp = reinterpret_cast<const vec_t *>(&input[j]);
                mm512_add_dpbusd_epi32(prod, inp[0], w[j / 64]);
            }
	    output[i] += _mm512_reduce_add_epi32(prod);
        }
        return;
    }
#endif
#ifdef AVX2
    assert(outputSize % 32 == 0 && inputSize % 32 == 0);
    std::memcpy(output, biases, outputSize * 4);
    for (unsigned i = 0; i < outputSize; i++) {
        __m256i prod = _mm256_setzero_si256();
        const __m256i *w = reinterpret_cast<const __m256i *>(weights[i]);
        for (unsigned j = 0; j < inputSize; j += 32) {
            const __m256i *inp = reinterpret_cast<const __m256i *>(&input[j]);
            mm256_add_dpbusd_epi32(prod, inp[0], w[j / 32]);
        }
        __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(prod),
                                    _mm256_extracti128_si256(prod, 1));
        sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x1b));
        output[i] += _mm_cvtsi128_si32(sum) + _mm_extract_epi32(sum, 1);
    }
#elif defined(SSSE3)
    const vec_t *inp = reinterpret_cast<const vec_t *>(input);
    const vec_t ones = _mm_set1_epi16(1);
    for (unsigned i = 0; i < outputSize; i++) {
        const vec_t *row = reinterpret_cast<const vec_t *>(weights + i);
        vec_t total  = _mm_setzero_si128();
        for (unsigned j = 0; j < chunks<uint8_t,simdWidth>(inputSize)/2; ++j) {
            vec_t p0 = _mm_madd_epi16(_mm_maddubs_epi16(inp[2*j+0], row[2*j+0]), ones);
            vec_t p1 = _mm_madd_epi16(_mm_maddubs_epi16(inp[2*j+1], row[2*j+1]), ones);
            vec_t sum = _mm_add_epi32(p0, p1);
            sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0xb));
            total = _mm_add_epi32(total,sum);
        }
#ifdef SSE41
        output[i] = _mm_cvtsi128_si32(total) + _mm_extract_epi32(total, 1) + biases[i];
#else
        total = _mm_add_epi32(total, _mm_shuffle_epi32(total, 1));
        output[i] = _mm_cvtsi128_si32(total) + biases[i];
#endif
    }
#elif defined(SSE2)
    const vec_t zeros = _mm_setzero_si128();
    const vec_t *inp = reinterpret_cast<const vec_t*>(input);
    for (unsigned i = 0; i < outputSize; i++) {
        __m128i sum_lo = _mm_cvtsi32_si128(biases[i]);
        __m128i sum_hi = zeros;
        const auto row = reinterpret_cast<const vec_t*>(&weights[i]);
        for (unsigned j = 0; j < chunks<uint8_t,simdWidth>(inputSize); ++j) {
            __m128i row_j = _mm_load_si128(&row[j]);
            __m128i input_j = _mm_load_si128(&inp[j]);
            __m128i row_signs = _mm_cmpgt_epi8(zeros, row_j);
            __m128i extended_row_lo = _mm_unpacklo_epi8(row_j, row_signs);
            __m128i extended_row_hi = _mm_unpackhi_epi8(row_j, row_signs);
            __m128i extended_input_lo = _mm_unpacklo_epi8(input_j, zeros);
            __m128i extended_input_hi = _mm_unpackhi_epi8(input_j, zeros);
            __m128i product_lo = _mm_madd_epi16(extended_row_lo, extended_input_lo);
            __m128i product_hi = _mm_madd_epi16(extended_row_hi, extended_input_hi);
            sum_lo = _mm_add_epi32(sum_lo, product_lo);
            sum_hi = _mm_add_epi32(sum_hi, product_hi);
        }
        __m128i sum = _mm_add_epi32(sum_lo, sum_hi);
        __m128i sum_high_64 = _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 0, 3, 2));
        sum = _mm_add_epi32(sum, sum_high_64);
        __m128i sum_second_32 = _mm_shufflelo_epi16(sum, _MM_SHUFFLE(1, 0, 3, 2));
        sum = _mm_add_epi32(sum, sum_second_32);
        output[i] = _mm_cvtsi128_si32(sum);
    }
#endif
}

template <size_t size, typename DataType>
inline void vec_copy(const DataType *in,DataType *out) {
    assert(in);
    assert(out);
    const vec_t *inp = reinterpret_cast<const vec_t *>(in);
    vec_t *outp = reinterpret_cast<vec_t *>(out);
    for (size_t i = 0; i < chunks<DataType,simdWidth>(size); ++i) {
#ifdef AVX512
        outp[i] = _mm512_load_si512(inp+i);
#elif defined(AVX2)
        outp[i] = _mm256_load_si256(inp+i);
#elif defined(SSE2) || defined(SSSE3)
        outp[i] = _mm_load_si128(inp+i);
#endif
    }
}

template <size_t size, typename InType, typename OutType>
inline void vec_add(const InType *in, OutType *out) {
    const vec_t *inp = reinterpret_cast<const vec_t *>(in);
    vec_t *outp = reinterpret_cast<vec_t *>(out);
    for (size_t i = 0; i < chunks<OutType,simdWidth>(size); ++i) {
#ifdef AVX512
        outp[i] = _mm512_add_epi16(outp[i], inp[i]);
#elif defined(AVX2)
        outp[i] = _mm256_add_epi16(outp[i], inp[i]);
#elif defined(SSE2) || defined(SSSE3)
        outp[i] = _mm_add_epi16(outp[i], inp[i]);
#endif
    }
}

template <size_t size, typename InType, typename OutType>
inline void vec_sub(const InType *in, OutType *out) {
    const vec_t *inp = reinterpret_cast<const vec_t *>(in);
    vec_t *outp = reinterpret_cast<vec_t *>(out);
    for (size_t i = 0; i < chunks<OutType,simdWidth>(size); ++i) {
#ifdef AVX512
        outp[i] = _mm512_sub_epi16(outp[i], inp[i]);
#elif defined(AVX2)
        outp[i] = _mm256_sub_epi16(outp[i], inp[i]);
#elif defined(SSE2) || defined(SSSE3)
        outp[i] = _mm_sub_epi16(outp[i], inp[i]);
#endif
    }
}

template <size_t size, typename InType, typename OutType>
inline void clamp(const InType *in, OutType *out, [[maybe_unused]] InType clampMax) {
    // TBD: can use AVX512 here?
#ifdef AVX2
    assert(sizeof(InType)==2);
    assert(sizeof(OutType)==1);
    const __m256i *inp = reinterpret_cast<const __m256i *>(in);
    __m256i *outp = reinterpret_cast<__m256i *>(out);
    const __m256i zero = _mm256_setzero_si256();
    for (size_t i = 0; i < chunks<OutType,256>(size); ++i) {
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
                0b11011000));
    }
#elif defined(SSE2) || defined(SSSE3)
    const vec_t *inp = reinterpret_cast<const vec_t *>(in);
    vec_t *outp = reinterpret_cast<vec_t *>(out);
    __m128i packedZeros = _mm_setzero_si128();
    __m128i packedMax = _mm_set1_epi16(clampMax);
    for (size_t i = 0; i < chunks<OutType,simdWidth>(size); ++i) {
        __m128i out0, out1;
        __m128i words0 = _mm_load_si128(
            reinterpret_cast<const __m128i *>(inp + 2 * i + 0));
        __m128i words1 = _mm_load_si128(
            reinterpret_cast<const __m128i *>(inp + 2 * i + 1));
        out0 = _mm_min_epi16(_mm_max_epi16(words0, packedZeros), packedMax);
        out1 = _mm_min_epi16(_mm_max_epi16(words1, packedZeros), packedMax);
        outp[i] = _mm_packs_epi16(out0,out1);
    }
#endif
}

template <size_t size, typename InType, typename OutType>
inline void scale_and_clamp(const InType *in, OutType *out, unsigned rshift, [[maybe_unused]] InType clampMax) {
#ifdef AVX2
    const __m256i *inp = reinterpret_cast<const __m256i *>(in);
    __m256i *outp = reinterpret_cast<__m256i *>(out);
    assert(sizeof(InType)==4);
    assert(sizeof(OutType)==1);
    const __m256i control = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
    const __m256i zero = _mm256_setzero_si256();
    for (size_t i = 0; i < chunks<OutType,256>(size); ++i) {
        // load 2x256 bit registers of shifted input data (32 bit input, 16 bit output)
        __m256i r1  = _mm256_srai_epi16(_mm256_packs_epi32(inp[4*i + 0],inp[4*i + 1]), rshift);
        __m256i r2  = _mm256_srai_epi16(_mm256_packs_epi32(inp[4*i + 2],inp[4*i + 3]), rshift);
        // clamp and store into one 256-bit output chunk
        outp[i] = _mm256_permutevar8x32_epi32(_mm256_max_epi8(_mm256_packs_epi16(r1, r2), zero), control);
    }
#elif defined(SSE2) || defined(SSSE3)
    const vec_t *inp = reinterpret_cast<const vec_t *>(in);
    vec_t *outp = reinterpret_cast<vec_t *>(out);
    assert(sizeof(InType)==4);
    assert(sizeof(OutType)==1);
#ifdef SSE41
    const vec_t zero = _mm_setzero_si128();
#else
    const vec_t k0x80s = _mm_set1_epi8(-128);
#endif
    for (size_t i = 0; i < chunks<OutType,simdWidth>(size); ++i) {
        // load 2x128 bit registers of shifted input data (32 bit input, 16 bit output) and clamp
        vec_t r1  = _mm_srai_epi16(_mm_packs_epi32(inp[4*i + 0],inp[4*i + 1]), rshift);
        vec_t r2  = _mm_srai_epi16(_mm_packs_epi32(inp[4*i + 2],inp[4*i + 3]), rshift);
        // pack into 8-bit output and clamp
        outp[i] =
#ifdef SSE41
            _mm_max_epi8(_mm_packs_epi16(r1, r2), zero);
#else
            _mm_subs_epi8(_mm_adds_epi8(_mm_packs_epi16(r1, r2), k0x80s), k0x80s);
#endif
    }
#endif
}

} // namespace simd

#endif
