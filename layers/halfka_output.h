// Copyright 2022 by Jon Dart. All Rights Reserved
#ifndef _NNUE_HALFKA_OUTPUT_H
#define _NNUE_HALFKA_OUTPUT_H

#include "typed.h"

// Transform the first layer output
template <typename InputType, typename OutputType, size_t size,
          size_t alignment = DEFAULT_ALIGN>
class HalfKaOutput
    : public TypedLayer<InputType, OutputType, size, size, alignment> {
  public:
    HalfKaOutput(int scaleFactor, int clampMax)
        : _scaleFactor(scaleFactor), _clampMax(clampMax) {}

    virtual ~HalfKaOutput() = default;

    virtual void doForward(const InputType *input, OutputType *output) const
        noexcept {
#if defined(SIMD)
        // TBD
        assert(0);

#else
        size_t offset = 0;
        for (size_t p = 0; p < 2; ++p, offset += size/2) {
            for (size_t i = 0; i < size/2; ++i) {
                OutputType sum0 = input[p][i];
                OutputType sum1 = input[p][i + size/2];
                sum0 = std::clamp<int>(sum0, 0, _clampMax);
                sum1 = std::clamp<int>(sum1, 0, _clampMax);
                output[offset + i] = static_cast<OutputType>((sum0 * sum1) >> _scaleFactor);
            }
        }
#endif
#ifdef NNUE_TRACE
        std::cout << "----" << std::endl;
        for (size_t i = 0; i < size; ++i) {
            std::cout << int(output[i]) << ' ';
        }
        std::cout << std::endl;
#endif
    }

  protected:
    int _scaleFactor, _clampMax;
};

#endif
