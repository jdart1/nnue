// Copyright 2022 by Jon Dart. All Rights Reserved
#ifndef _NNUE_HALFKA_OUTPUT_H
#define _NNUE_HALFKA_OUTPUT_H

#include "typed.h"

// Transform the accumulator into first layer output
template <typename InputType, typename AccumulatorType, typename OutputType, size_t size,
          size_t alignment = DEFAULT_ALIGN>
class HalfKaOutput
    : public TypedLayer<InputType, OutputType, size, size, alignment> {
  public:
    HalfKaOutput(int scaleFactor, int clampMax)
        : _scaleFactor(scaleFactor), _clampMax(clampMax) {}

    virtual ~HalfKaOutput() = default;

    virtual void doForward([[maybe_unused]] const InputType *input, [[maybe_unused]] OutputType *output) const noexcept {
        // no-op for this layer: use method below
        assert(0);
    }

    void postProcessAccum(const AccumulatorType &accum, OutputType *output) const {
#ifdef SIMD
        simd::multAndSum<InputType,OutputType,size/2>(accum.getOutput(AccumulatorHalf::Lower),output,_clampMax,_scaleFactor);
        simd::multAndSum<InputType,OutputType,size/2>(accum.getOutput(AccumulatorHalf::Upper),output + size/2,_clampMax,_scaleFactor);
#else
        size_t offset = 0;
        static const AccumulatorHalf halves[2] = {AccumulatorHalf::Lower, AccumulatorHalf::Upper};
        for (size_t p = 0; p < 2; ++p, offset += size/2) {
            const InputType *input = accum.getOutput(halves[p]);
            for (size_t i = 0; i < size/2; ++i) {
                InputType sum0 = input[i];
                InputType sum1 = input[i + size/2];
                sum0 = std::clamp<int>(sum0, 0, _clampMax);
                sum1 = std::clamp<int>(sum1, 0, _clampMax);
                output[offset + i] = static_cast<OutputType>((sum0 * sum1) >> _scaleFactor);
            }
        }
#endif
#ifdef NNUE_TRACE
        std::cout << "---- halfka_output " << std::endl;
        for (size_t i = 0; i < size; ++i) {
            std::cout << int(output[i]) << ' ';
            if ((i+1) % 64 == 0) std::cout << std::endl;
        }
        std::cout << std::endl;
#endif
    }

  protected:
    int _scaleFactor, _clampMax;
};

#endif
