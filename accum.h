// Copyright 2021 by Jon Dart. All Rigths Reserved.
#ifndef _NNUE_ACCUM_H
#define _NNUE_ACCUM_H

#include "chess.h"
#include "nndefs.h"

enum class AccumulatorHalf { Lower, Upper };

enum class AccumulatorState { Empty, Computed };

// Holds calculations that reprepsent the output of the first layer, pre-scaling
template <typename OutputType, typename WeightType, typename BiasType,
          size_t size, size_t alignment = DEFAULT_ALIGN>
class Accumulator {
  public:
    Accumulator() { _states[0] = _states[1] = AccumulatorState::Empty; }

    virtual ~Accumulator() = default;

    static AccumulatorHalf getHalf(Color c, Color sideToMove) {
        // TBD direct enum compare not working?
        int c1 = (c == White) ? 0 : 1;
        int c2 = (sideToMove == White) ? 0 : 1;
        return c1 == c2 ? AccumulatorHalf::Lower : AccumulatorHalf::Upper;
    }

    void init(const BiasType *data) {
        OutputType *out = _accum;
#ifdef SIMD
        if constexpr (size*16 % simd::simdWidth == 0 && sizeof(WeightType) == sizeof(OutputType) && sizeof(WeightType)==2) {
            simd::vec_copy<size,OutputType>(data,out);
        }
        else
#endif
        for (size_t i = 0; i < size; ++i) {
            *out++ = *data++;
        }
    }

    void init_half(AccumulatorHalf half, const BiasType *data) {
        const OutputType *in = data;
        OutputType *out = _accum + offset(half);
#ifdef SIMD
        if constexpr ((size/2)*16 % simd::simdWidth == 0 && sizeof(WeightType) == sizeof(OutputType) && sizeof(WeightType)==2) {
            simd::vec_copy<size/2,OutputType>(in,out);
        }
        else
#endif
        for (size_t i = 0; i < size / 2; ++i) {
            *out++ = static_cast<OutputType>(*in++);
        }
    }

    void copy_half(AccumulatorHalf half,
                   const Accumulator<OutputType, WeightType, BiasType, size,
                                     alignment> &source,
                   AccumulatorHalf sourceHalf) {
        const OutputType *in = source._accum + offset(sourceHalf);
        OutputType *out = _accum + offset(half);
#ifdef SIMD
        if constexpr ((size/2)*16 % simd::simdWidth == 0 && sizeof(WeightType) == sizeof(OutputType) && sizeof(WeightType)==2) {
            simd::vec_copy<size/2,OutputType>(in,out);
        }
        else
#endif
        for (size_t i = 0; i < size / 2; ++i) {
            *out++ = static_cast<OutputType>(*in++);
        }
    }

    // Update half of the accumulator
    void add_half(AccumulatorHalf half, const WeightType *data) {
        const OutputType *in = data;
        OutputType *out = _accum + offset(half);
#ifdef SIMD
        if constexpr ((size/2)*16 % simd::simdWidth == 0 && sizeof(WeightType) == sizeof(OutputType) && sizeof(WeightType)==2) {
            simd::vec_add<size/2,WeightType,OutputType>(in,out);
        }
        else
#endif
        for (size_t i = 0; i < size / 2; ++i) {
            *out++ += static_cast<OutputType>(*in++);
        }
    }

    // Update half of the accumulator
    void sub_half(AccumulatorHalf half, const WeightType *data) {
        const OutputType *in = data;
        OutputType *out = _accum + offset(half);
#ifdef SIMD
        if constexpr ((size/2)*16 % simd::simdWidth == 0 && sizeof(WeightType) == sizeof(OutputType) && sizeof(WeightType)==2) {
            simd::vec_sub<size/2,WeightType,OutputType>(in,out);
        }
        else
#endif
        for (size_t i = 0; i < size / 2; ++i) {
            *out++ -= static_cast<OutputType>(*in++);
        }
    }

    const OutputType *getOutput() const noexcept { return _accum; }

    const OutputType *getOutput(AccumulatorHalf half) const noexcept {
        return _accum + offset(half);
    }

    AccumulatorState getState(AccumulatorHalf half) const noexcept {
        return _states[half == AccumulatorHalf::Lower ? 0 : 1];
    }

    void setState(AccumulatorHalf half, AccumulatorState state) {
        _states[half == AccumulatorHalf::Lower ? 0 : 1] = state;
    }

    void setState(AccumulatorState state) { _states[0] = _states[1] = state; }

    void setEmpty() {
        setState(AccumulatorHalf::Lower, AccumulatorState::Empty);
        setState(AccumulatorHalf::Upper, AccumulatorState::Empty);
    }

    size_t getSize() const noexcept {
        return size;
    }

  private:
    size_t offset(AccumulatorHalf half) const noexcept {
        return (half == AccumulatorHalf::Lower ? 0 : 1) * size / 2;
    }

    alignas(alignment) OutputType _accum[size];
    AccumulatorState _states[2];
};

#endif
