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
class LinearLayer : public TypedLayer<InputType, OutputType, inputSize,
                                      outputSize, alignment> {
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
#if defined(SIMD)
        if constexpr (outputSize == 1) { // output layer
            simd::dotProduct32x1(input,_weights[0],_biases,output);
        }
        else if constexpr (outputSize == 32) {
            simd::dotProductnx32<inputSize,outputSize>(input,_weights,_biases,output);
        }
        else
#endif
        {
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

    virtual const WeightType *getCol(size_t col) const noexcept {
        return _weights[col];
    }

    virtual void setCol(size_t index, const WeightType *col) {
        for (size_t i = 0; i < inputSize; ++i)
            _weights[index][i] = col[i];
    }

  private:
    alignas(alignment) BiasType _biases[outputSize];
    alignas(alignment) WeightType _weights[outputSize][inputSize];
};

#endif
