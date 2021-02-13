// Copyright 2021 by Jon Dart. All Rights Reserved
#ifndef _NNUE_SCALE_CLAMP_H
#define _NNUE_SCALE_CLAMP_H

#include "typed.h"

template <typename InputType, typename OutputType, size_t size>
class ScaleAndClamp : public TypedLayer<InputType, OutputType, size, size>
{
public:
    ScaleAndClamp(int scaleFactor, int clampMax) :
        _scaleFactor(scaleFactor), _clampMax(clampMax) 
        {
        }
    
    virtual ~ScaleAndClamp() = default;

    virtual void doForward(const InputType *input, OutputType *output) const noexcept {
        for (size_t i = 0; i < size; i++) {
            *output++ = static_cast<OutputType>(std::clamp(static_cast<int>(input[i]/_scaleFactor),0,_clampMax));
        }
    }

protected:
    int _scaleFactor, _clampMax;
};
    
#endif
