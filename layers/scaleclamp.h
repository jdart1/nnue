// Copyright 2021, 2022 by Jon Dart. All Rights Reserved
#ifndef _NNUE_SCALE_CLAMP_H
#define _NNUE_SCALE_CLAMP_H

#include "typed.h"

template <typename InputType, typename OutputType, size_t size,
          size_t alignment = DEFAULT_ALIGN>
class ScaleAndClamp
    : public TypedLayer<InputType, OutputType, size, size, alignment> {
  public:
    // scaleFactor is right shift, clampMax is upper limit for output
    ScaleAndClamp(int scaleFactor, int clampMax)
        : _scaleFactor(scaleFactor), _clampMax(clampMax) {}

    virtual ~ScaleAndClamp() = default;

    virtual void doForward(const InputType *input, OutputType *output) const
        noexcept {
#if defined(SIMD)
        simd::scale_and_clamp<size, InputType, OutputType>(input, output, _scaleFactor,
                                                           _clampMax);
#else
        for (size_t i = 0; i < size; i++) {
            *output++ = static_cast<OutputType>(
                                                std::clamp<InputType>(input[i] >> _scaleFactor, 0, _clampMax));
        }
#endif
    }

  protected:
    int _scaleFactor, _clampMax;
};

#endif
