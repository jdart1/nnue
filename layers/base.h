// Copyright 2021 by Jon Dart. All Rights Reserved.
#ifndef _NNUE_BASE_LAYER_H
#define _NNUE_BASE_LAYER_H

// base layer class
class BaseLayer {
  public:
    BaseLayer() = default;

    virtual ~BaseLayer() = default;

    // propagate data through the layer (typeless version)
    virtual void forward(const void *input, void *output) const noexcept = 0;

    virtual size_t getInputSize() const noexcept = 0;

    virtual size_t getOutputSize() const noexcept = 0;

    virtual size_t bufferSize() const noexcept = 0;

    virtual std::istream &read(std::istream &s) { return s; }

    virtual void zero() {}
};

#endif
