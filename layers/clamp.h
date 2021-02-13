// Copyright 2021 by Jon Dart. All Rights Reserved
#ifndef _NNUE_CLAMP_H
#define _NNUE_CLAMP_H

#include "typed.h"

template <typename InputType, typename OutputType, size_t size>
class Clamp : public TypedLayer<InputType, OutputType, size, size> {
  public:
    Clamp(OutputType clampMax) : _clampMax(clampMax) {}

    virtual ~Clamp() = default;

    virtual void doForward(const InputType *input, OutputType *output) const
        noexcept {
        for (size_t i = 0; i < size; i++) {
            *output++ = static_cast<OutputType>(std::clamp<InputType>(
                input[i], 0, static_cast<InputType>(_clampMax)));
        }
    }

  protected:
    OutputType _clampMax;
};

#endif
