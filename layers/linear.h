// Copyright 2021 by Jon Dart. All Rights Reserved.
#ifndef _NNUE_LINEAR_H
#define _NNUE_LINEAR_H

#include "nndefs.h"
#include "typed.h"

// This class defines a linear transformation layer of the NNUE.
//
template <typename InputType, typename WeightType, typename BiasType,
          typename OutputType, size_t inputSize, size_t outputSize,
          size_t alignment = DEFAULT_ALIGN>
class LinearLayer
    : public TypedLayer<InputType, OutputType, inputSize, outputSize> {
  public:
    LinearLayer() = default;

    virtual ~LinearLayer() = default;

    // propagate data through the layer
    virtual inline void doForward(const InputType *input,
                                  OutputType *output) const noexcept {
        dotProduct(input, output);
    }

    inline void dotProduct(const InputType *input, OutputType *output) const
        noexcept {
        // generic implementation
        for (size_t i = 0; i < outputSize; i++) {
            output[i] = static_cast<OutputType>(this->_biases[i]);
        }
        for (size_t i = 0; i < outputSize; i++) {
            for (size_t j = 0; j < inputSize; j++) {
                output[i] +=
                    static_cast<OutputType>(input[j] * this->_weights[i][j]);
            }
        }
    }

    virtual void zero() {
        for (size_t i = 0; i < outputSize; ++i) {
            _biases[i] = 0;
        }
        for (size_t i = 0; i < outputSize; ++i) {
            for (size_t j = 0; j < inputSize; ++j) {
                _weights[i][j] = 0;
            }
        }
    }

    virtual std::istream &read(std::istream &s) {
        // Note: linear layers are stored in column order
        for (size_t i = 0; i < outputSize && s.good(); ++i) {
            _biases[i] = read_little_endian<BiasType>(s);
        }
        for (size_t i = 0; i < outputSize && s.good(); ++i) {
            for (size_t j = 0; j < inputSize && s.good(); ++j) {
                _weights[i][j] = read_little_endian<WeightType>(s);
            }
        }
        return s;
    }

    virtual const BiasType *getBiases() const noexcept { return _biases; }

    virtual const WeightType *getCol(size_t row) const noexcept {
        return _weights[row];
    }

    virtual void setCol(size_t row, const WeightType *col) {
        for (size_t i = 0; i < inputSize; ++i)
            _weights[row][i] = col[i];
    }

  private:
    alignas(alignment) BiasType _biases[outputSize];
    alignas(alignment) WeightType _weights[outputSize][inputSize];
};

#endif
