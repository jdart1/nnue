// Copyright 2021, 2022 by Jon Dart. All Rights Reserved
#ifndef _NNUE_SCALE_CLAMP_H
#define _NNUE_SCALE_CLAMP_H

#include "typed.h"

template <typename InputType, typename OutputType, size_t size,
          size_t alignment = DEFAULT_ALIGN>
class ScaleAndClamp
    : public TypedLayer<InputType, OutputType, size, size, alignment> {
  public:
    ScaleAndClamp(int scaleFactor, int clampMax)
        : _scaleFactor(scaleFactor), _clampMax(clampMax) { assert(scaleFactor); }

    virtual ~ScaleAndClamp() = default;

    virtual void doForward(const InputType *input, OutputType *output) const
        noexcept {
#if defined(SIMD)
        simd::scale_and_clamp<size, InputType, OutputType>(input, output, _scaleFactor,
                                                           _clampMax);
#else
        for (size_t i = 0; i < size; i++) {
            output[i] = static_cast<OutputType>(
                std::clamp<InputType>(input[i] >> _scaleFactor, 0, _clampMax));
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
