// Copyright 2021 by Jon Dart. All Rights Reserved.
#ifndef _NNUE_TYPED_LAYER_H
#define _NNUE_TYPED_LAYER_H

#include "base.h"
#include "util.h"

// typed layer
template <typename InputType, typename OutputType, size_t inputSize,
          size_t outputSize, size_t alignment>
class TypedLayer : public BaseLayer {
  public:
    TypedLayer() = default;

    virtual ~TypedLayer() = default;

    // propagate data through the layer
    virtual void forward(const void *input, void *output) const noexcept {
        // delegate to typed pointer version
        doForward(static_cast<const InputType *>(input),
                  static_cast<OutputType *>(output));
    }

    virtual void doForward(const InputType *input, OutputType *output) const
        noexcept = 0;

    virtual size_t getInputSize() const noexcept { return inputSize; }

    virtual size_t getOutputSize() const noexcept { return outputSize; }

    virtual size_t bufferSize() const noexcept {
        const size_t size = outputSize*sizeof(OutputType);
        if (size % alignment != 0) {
            assert((size + alignment - (size % alignment)) % alignment == 0);
            return size + alignment - (size % alignment);
        }
        else
            return size;
    }

    virtual std::istream &read(std::istream &s) { return s; }
};

#endif
