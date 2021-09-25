// Copyright 2021 by Jon Dart. All Rights Reserved
#ifndef _NNUE_CLAMP_H
#define _NNUE_CLAMP_H

#include "typed.h"

template <typename InputType, typename OutputType, size_t size,
          size_t alignment = DEFAULT_ALIGN>
class Clamp : public TypedLayer<InputType, OutputType, size, size, alignment> {
  public:
    Clamp(OutputType clampMax) : _clampMax(clampMax) {}

    virtual ~Clamp() = default;

    virtual void doForward(const InputType *input, OutputType *output) const
        noexcept {
#if defined(SIMD)
        simd::clamp<size, InputType, OutputType>(input, output, _clampMax);
#else
        for (size_t i = 0; i < size; i++) {
            *output++ = static_cast<OutputType>(std::clamp<InputType>(
                input[i], 0, static_cast<InputType>(_clampMax)));
        }
#endif
    }

  protected:
    OutputType _clampMax;
};

#endif
