// Copyright 2021-2024 by Jon Dart. All Rigths Reserved.
#ifndef _NNUE_ACCUM_H
#define _NNUE_ACCUM_H

#include "chess.h"
#include "nndefs.h"

#include <cstring>

enum class AccumulatorHalf { Lower, Upper };

enum class AccumulatorState { Empty, Computed };

// Holds calculations that reprepsent the output of the first layer, pre-scaling
template <typename OutputType, typename WeightType, typename BiasType, size_t size,
          size_t alignment = DEFAULT_ALIGN>
class Accumulator {
  public:
    typedef const OutputType (*OutputPtr)[size];

    Accumulator() { _states[0] = _states[1] = AccumulatorState::Empty; }

    virtual ~Accumulator() = default;

    static AccumulatorHalf getHalf(Color c, Color sideToMove) {
        // TBD direct enum compare not working?
        int c1 = (c == White) ? 0 : 1;
        int c2 = (sideToMove == White) ? 0 : 1;
        return c1 == c2 ? AccumulatorHalf::Lower : AccumulatorHalf::Upper;
    }

    void init(const BiasType *data) {
        init_half(AccumulatorHalf::Lower, data);
        init_half(AccumulatorHalf::Upper, data);
    }

    void init_half(AccumulatorHalf half, const BiasType *data) {
        OutputType *out = _accum[halfToIndex(half)];
        if constexpr (sizeof(OutputType) == sizeof(BiasType)) {
            std::memcpy(out, data, sizeof(OutputType) * size);
        } else {
            for (size_t i = 0; i < size; ++i) {
                *out++ = *data++;
            }
        }
    }

    void copy_half(AccumulatorHalf half,
                   const Accumulator<OutputType, WeightType, BiasType, size, alignment> &source,
                   AccumulatorHalf sourceHalf) {
        const WeightType *in = source._accum[halfToIndex(sourceHalf)];
        OutputType *out = _accum[halfToIndex(half)];
        if constexpr (sizeof(OutputType) == sizeof(WeightType)) {
            std::memcpy(out, in, sizeof(OutputType) * size);
        } else {
            for (size_t i = 0; i < size; ++i) {
                *out++ = static_cast<OutputType>(*in++);
            }
        }
    }

    // Update half of the accumulator (does not use SIMD)
    void add_half(AccumulatorHalf half, const WeightType *data) {
        const WeightType *in = data;
        OutputType *out = _accum[halfToIndex(half)];
        for (size_t i = 0; i < size; ++i) {
            *out++ += static_cast<OutputType>(*in++);
        }
    }

    // Update half of the accumulator (does not use SIMD)
    void sub_half(AccumulatorHalf half, const WeightType *data) {
        const OutputType *in = data;
        OutputType *out = _accum[halfToIndex(half)];
        for (size_t i = 0; i < size; ++i) {
            *out++ -= static_cast<OutputType>(*in++);
        }
    }

    OutputPtr getOutput() const noexcept { return _accum; }

    const OutputType *getOutput(AccumulatorHalf half) const noexcept {
        return _accum[halfToIndex(half)];
    }

    OutputType *data(AccumulatorHalf half) { return _accum[halfToIndex(half)]; }

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

    size_t getSize() const noexcept { return size; }

#ifdef NNUE_TRACE
    template <typename _OutputType, typename _WeightType, typename _BiasType, size_t _size,
              size_t _alignment>
    friend std::ostream &
    operator<<(std::ostream &o,
               const Accumulator<_OutputType, _WeightType, _BiasType, _size, _alignment> &accum);
#endif

    template <typename _OutputType, typename _WeightType, typename _BiasType, size_t _size,
              size_t _alignment>
    friend bool
    operator==(const Accumulator<_OutputType, _WeightType, _BiasType, _size, _alignment> &accum1,
               const Accumulator<_OutputType, _WeightType, _BiasType, _size, _alignment> &accum2);

    template <typename _OutputType, typename _WeightType, typename _BiasType, size_t _size,
              size_t _alignment>
    friend bool
    operator!=(const Accumulator<_OutputType, _WeightType, _BiasType, _size, _alignment> &accum1,
               const Accumulator<_OutputType, _WeightType, _BiasType, _size, _alignment> &accum2);

  private:
    static inline unsigned halfToIndex(AccumulatorHalf half) {
        return (half == AccumulatorHalf::Lower ? 0 : 1);
    }

    alignas(alignment) OutputType _accum[2][size];
    AccumulatorState _states[2];
};

#ifdef NNUE_TRACE
template <typename OutputType, typename WeightType, typename BiasType, size_t size,
          size_t alignment = DEFAULT_ALIGN>
inline std::ostream &operator<<(std::ostream &o,
                                const Accumulator<OutputType, WeightType, size, alignment> &accum) {
    for (unsigned p = 0; p < 2; ++p) {
        std::cout << "perspective " << p << std::endl;
        for (unsigned i = 0; i < size; ++i) {
            std::cout << static_cast<int>(accum._accum[p][i]) << ' ';
            if ((i + 1) % 64 == 0)
                std::cout << std::endl;
        }
    }
    return o;
}
#endif

template <typename _OutputType, typename _WeightType, typename _BiasType, size_t _size,
          size_t _alignment>
inline bool
operator==(const Accumulator<_OutputType, _WeightType, _BiasType, _size, _alignment> &accum1,
           const Accumulator<_OutputType, _WeightType, _BiasType, _size, _alignment> &accum2) {
    const auto p = accum1._accum;
    const auto q = accum2._accum;
    for (size_t h = 0; h < 2; ++h) {
        for (size_t i = 0; i < _size; ++i) {
            if (p[h][i] != q[h][i]) {
                return false;
            }
        }
    }
    return true;
}

template <typename _OutputType, typename _WeightType, typename _BiasType, size_t _size,
          size_t _alignment>
inline bool
operator!=(const Accumulator<_OutputType, _WeightType, _BiasType, _size, _alignment> &accum1,
           const Accumulator<_OutputType, _WeightType, _BiasType, _size, _alignment> &accum2) {
    return !(accum1 == accum2);
}

#endif
