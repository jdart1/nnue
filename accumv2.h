// Copyright 2021, 2022 by Jon Dart. All Rigths Reserved.
#ifndef _NNUE_ACCUM_V2_H
#define _NNUE_ACCUM_V2_H

#include "chess.h"
#include "nndefs.h"

#include <cstring>

// Accumulator with Piece-Square buckets

enum class AccumulatorHalf { Lower, Upper };

enum class AccumulatorState { Empty, Computed };

// Holds calculations that reprepsent the output of the first layer, pre-scaling
template <typename OutputType, typename WeightType, typename BiasType,
          typename PSQType,
          size_t size, size_t psq_buckets, size_t alignment = DEFAULT_ALIGN>
class AccumulatorV2 {
  public:

    typedef const OutputType (*OutputPtr) [size];
    
    AccumulatorV2() { _states[0] = _states[1] = AccumulatorState::Empty; }

    virtual ~AccumulatorV2() = default;

    static AccumulatorHalf getHalf(Color c, Color sideToMove) {
        // TBD direct enum compare not working?
        int c1 = (c == White) ? 0 : 1;
        int c2 = (sideToMove == White) ? 0 : 1;
        return c1 == c2 ? AccumulatorHalf::Lower : AccumulatorHalf::Upper;
    }

    void init(const BiasType *data) {
        for (unsigned p = 0; p < 2; ++p) {
            OutputType *out = _accum[p];
#ifdef SIMD
            simd::vec_copy<size,OutputType>(data,out);
#else
            for (size_t i = 0; i < size; ++i) {
                out[i] = *data[i];
            }
#endif        
        }
        std::memset(_psq_accum[0],'\0',sizeof(PSQType)*psq_buckets);
        std::memset(_psq_accum[1],'\0',sizeof(PSQType)*psq_buckets);
    }

    void init_half(AccumulatorHalf half, const BiasType *data) {
        const OutputType *in = data;
        OutputType *out = _accum[halfToIndex(half)];
#ifdef SIMD
        simd::vec_copy<size,OutputType>(in,out);
#else
        for (size_t i = 0; i < size; ++i) {
            *out++ = static_cast<OutputType>(*in++);
        }
#endif
        std::memset(_psq_accum[halfToIndex(half)],'\0',sizeof(PSQType)*psq_buckets);
    }

    void copy_half(AccumulatorHalf half,
                   const AccumulatorV2<OutputType, WeightType, BiasType, PSQType, size,
                                     psq_buckets,
                                     alignment> &source,
                   AccumulatorHalf sourceHalf) {
        const OutputType *in = source._accum[halfToIndex(sourceHalf)];
        OutputType *out = _accum[halfToIndex(half)];
#ifdef SIMD
        simd::vec_copy<size,OutputType>(in,out);
#else
        for (size_t i = 0; i < size; ++i) {
            *out++ = static_cast<OutputType>(*in++);
        }
#endif
        std::memcpy(_psq_accum[halfToIndex(half)],source._psq_accum[halfToIndex(sourceHalf)],sizeof(PSQType)*psq_buckets);
    }

    // Update half of the accumulator
    void add_half(AccumulatorHalf half, const WeightType *data, const PSQType *psqData) {
        const OutputType *in = data;
        OutputType *out = _accum[halfToIndex(half)];
        const PSQType *psq_in = psqData;
        PSQType *psq_out = _psq_accum[halfToIndex(half)];
#ifdef SIMD
        simd::vec_add<size,WeightType,OutputType>(in,out);
        simd::vec_add<PSQBuckets,PSQType,PSQType>(psq_in,psq_out);
#else
        for (size_t i = 0; i < size; ++i) {
            *out++ += static_cast<OutputType>(*in++);
        }
        for (size_t i = 0; i < psq_buckets; ++i) {
            *psq_out++ += static_cast<OutputType>(*psq_in++);
        }
#endif
    }

    // Update half of the accumulator
    void sub_half(AccumulatorHalf half, const WeightType *data, const PSQType *psqData) {
        const OutputType *in = data;
        OutputType *out = _accum[halfToIndex(half)];
        const PSQType *psq_in = psqData;
        PSQType *psq_out = _psq_accum[halfToIndex(half)];
#ifdef SIMD
        simd::vec_sub<size,WeightType,OutputType>(in,out);
        simd::vec_sub<PSQBuckets,PSQType,PSQType>(psq_in,psq_out);
#else
        for (size_t i = 0; i < size; ++i) {
            *out++ -= static_cast<OutputType>(*in++);
        }
        for (size_t i = 0; i < psq_buckets; ++i) {
            *psq_out++ -= static_cast<OutputType>(*psq_in++);
        }
#endif
    }

    OutputPtr getOutput() const noexcept { return _accum; }

    const OutputType *getOutput(AccumulatorHalf half) const noexcept { return _accum[halfToIndex(half)]; }

    const PSQType *getPSQ(AccumulatorHalf half) const noexcept {
        return _psq_accum[halfToIndex(half)];
    }

    AccumulatorState getState(AccumulatorHalf half) const noexcept {
        return _states[halfToIndex(half)];
    }

    void setState(AccumulatorHalf half, AccumulatorState state) {
        _states[halfToIndex(half)] = state;
    }

    void setState(AccumulatorState state) { _states[0] = _states[1] = state; }

    void setEmpty() {
        setState(AccumulatorHalf::Lower, AccumulatorState::Empty);
        setState(AccumulatorHalf::Upper, AccumulatorState::Empty);
    }

    size_t getSize() const noexcept {
        return size;
    }

#ifdef NNUE_TRACE
    template <typename _OutputType, typename _WeightType, typename _BiasType,
              typename _PSQType, size_t _size, size_t _psq_buckets, size_t _alignment>
    friend std::ostream & operator << (std::ostream &o, const AccumulatorV2<_OutputType,
                                       _WeightType, _BiasType, _PSQType, _size,
                                       _psq_buckets, _alignment> &accum);
#endif    

    template <typename _OutputType, typename _WeightType, typename _BiasType,
              typename _PSQType, size_t _size, size_t _psq_buckets, size_t _alignment>
    friend bool operator == (const AccumulatorV2<_OutputType,
                             _WeightType, _BiasType, _PSQType, _size,
                             _psq_buckets, _alignment> &accum1, const AccumulatorV2<_OutputType,
                             _WeightType, _BiasType, _PSQType, _size,
                             _psq_buckets, _alignment> &accum2);

    template <typename _OutputType, typename _WeightType, typename _BiasType,
              typename _PSQType, size_t _size, size_t _psq_buckets, size_t _alignment>
    friend bool operator != (const AccumulatorV2<_OutputType,
                             _WeightType, _BiasType, _PSQType, _size,
                             _psq_buckets, _alignment> &accum1, const AccumulatorV2<_OutputType,
                             _WeightType, _BiasType, _PSQType, _size,
                             _psq_buckets, _alignment> &accum2);
private:
    static inline unsigned halfToIndex(AccumulatorHalf half) {
        return (half == AccumulatorHalf::Lower ? 0 : 1);
    }

    alignas(alignment) OutputType _accum[2][size];
    alignas(alignment) PSQType _psq_accum[2][psq_buckets];
    AccumulatorState _states[2];
};

#ifdef NNUE_TRACE
template <typename OutputType, typename WeightType, typename BiasType,
          typename PSQType,
          size_t size, size_t psq_buckets, size_t alignment = DEFAULT_ALIGN>
inline std::ostream & operator << (std::ostream &o, const AccumulatorV2<OutputType,
                            WeightType, BiasType, PSQType, size,
                            psq_buckets, alignment> &accum) {
    std::cout << std::endl << "-----" << std::endl;
    for (unsigned p = 0; p < 2; ++p) {
        std::cout << "perspective " << p << std::endl;
        for (unsigned i = 0; i < size; ++i) {
            std::cout << static_cast<int>(accum._accum[p][i]) << ' ';
            if ((i+1)%64==0) std::cout << std::endl;
        }
    }
    std::cout << std::endl << "psq: " << std::endl;
    for (unsigned p = 0; p < 2; ++p) {
        for (unsigned i = 0; i<PSQBuckets; ++i) {
            std::cout << static_cast<int>(accum._psq_accum[p][i]) << ' ';
        }
        std::cout << std::endl;
    }
    return o;
}
#endif

template <typename _OutputType, typename _WeightType, typename _BiasType,
          typename _PSQType, size_t _size, size_t _psq_buckets, size_t _alignment>
inline bool operator == (const AccumulatorV2<_OutputType,
                  _WeightType, _BiasType, _PSQType, _size,
                  _psq_buckets, _alignment> &accum1, const AccumulatorV2<_OutputType,
                  _WeightType, _BiasType, _PSQType, _size,
                  _psq_buckets, _alignment> &accum2) {
    const auto p = accum1._accum;
    const auto q = accum2._accum;
    for (size_t h = 0; h < 2; ++h) {
        for (size_t i = 0; i < _size; ++i) {
            if (p[h][i] != q[h][i]) {
                return false;
            }
        }
        for (size_t i = 0; i < PSQBuckets; i++) {
            if (accum1._psq_accum[h][i] != accum2._psq_accum[h][i]) {
                return false;
            }
        }
    }
    return true;
}

template <typename _OutputType, typename _WeightType, typename _BiasType,
          typename _PSQType, size_t _size, size_t _psq_buckets, size_t _alignment>
inline bool operator != (const AccumulatorV2<_OutputType,
                  _WeightType, _BiasType, _PSQType, _size,
                  _psq_buckets, _alignment> &accum1, const AccumulatorV2<_OutputType,
                  _WeightType, _BiasType, _PSQType, _size,
                  _psq_buckets, _alignment> &accum2) {
    return !(accum1 == accum2);
}

#endif
