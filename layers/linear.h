// Copyright 2021-2022, 2024 by Jon Dart. All Rights Reserved.
#ifndef _NNUE_LINEAR_H
#define _NNUE_LINEAR_H

#include "nndefs.h"
#include "typed.h"

// This class defines a linear transformation layer of the NNUE.
//
template <typename InputType, typename WeightType, typename BiasType, typename OutputType,
          size_t inputSize, size_t outputSize, size_t alignment = DEFAULT_ALIGN>
class LinearLayer : public TypedLayer<InputType, OutputType, inputSize, outputSize, alignment> {

    static constexpr size_t roundedInputSize = std::max<size_t>(32, inputSize);

  public:
    LinearLayer() : _biases{{0}}, _weights{{0}} {
    }

    virtual ~LinearLayer() = default;

    // propagate data through the layer
    virtual inline void doForward(const InputType *input, OutputType *output) const noexcept {
        dotProduct(input, output);
#ifdef NNUE_TRACE
        std::cout << "---- linear " << inputSize << 'x' << outputSize << " ----" << std::endl;
        for (unsigned i = 0; i < outputSize; ++i) {
            std::cout << int(output[i]) << ' ';
        }
        std::cout << std::endl;
#endif
    }

    inline void dotProduct(const InputType *input, OutputType *output) const noexcept {
#if defined(SIMD)
        if constexpr (outputSize == 1 && sizeof(WeightType) == 1) { // output layer
            simd::dotProduct32x1(input, _weights[0], _biases, output);
        } else if constexpr (inputSize >= 32 && sizeof(WeightType) == 1) {
            simd::dotProductnxn<inputSize, roundedInputSize, outputSize>(input, _weights, _biases,
                                                                         output);
        } else
#endif
        {
            // generic implementation
            for (size_t i = 0; i < outputSize; i++) {
                output[i] = static_cast<OutputType>(this->_biases[i]);
            }
            for (size_t i = 0; i < outputSize; i++) {
                for (size_t j = 0; j < inputSize; j++) {
                    output[i] += static_cast<OutputType>(input[j] * this->_weights[i][j]);
                }
            }
        }
    }

    virtual void zero() {
        for (size_t i = 0; i < outputSize; ++i) {
            _biases[i] = 0;
        }
        for (size_t i = 0; i < outputSize; ++i) {
            for (size_t j = 0; j < roundedInputSize; ++j) {
                _weights[i][j] = 0;
            }
        }
    }

    virtual std::istream &read(std::istream &s) {
#ifdef STOCKFISH_FORMAT
        // Note: linear layers are stored in column order
        for (size_t i = 0; i < outputSize && s.good(); ++i) {
            _biases[i] = read_little_endian<BiasType>(s);
        }
        for (size_t i = 0; i < outputSize && s.good(); ++i) {
            for (size_t j = 0; j < roundedInputSize && s.good(); ++j) {
                _weights[i][j] = read_little_endian<WeightType>(s);
            }
        }
#else
        for (size_t i = 0; i < inputSize && s.good(); ++i) {
            for (size_t j = 0; j < outputSize && s.good(); ++j) {
                // flip rows and columns for easier computation
                _weights[j][i] = read_little_endian<WeightType>(s);
            }
        }
        for (size_t i = 0; i < outputSize && s.good(); ++i) {
            _biases[i] = read_little_endian<BiasType>(s);
        }
#endif
        return s;
    }

    virtual std::istream &readWeights(std::istream &s) {
#ifdef NNUE_TRACE
        int min_weight = 1<<30, max_weight = -(1<<30);
#endif
        for (size_t i = 0; i < inputSize && s.good(); ++i) {
            for (size_t j = 0; j < outputSize && s.good(); ++j) {
                // flip rows and columns for easier computation
                _weights[j][i] = read_little_endian<WeightType>(s);
#ifdef NNUE_TRACE
                if (_weights[j][i] < min_weight) min_weight = _weights[j][i];
                if (_weights[j][i] > max_weight) max_weight = _weights[j][i];
#endif
            }
        }
#ifdef NNUE_TRACE
        std::cout << "min output weight = " << min_weight << " max output weight = " << max_weight << std::endl;
#endif
        return s;
    }

    virtual std::istream &readBiases(std::istream &s) {
#ifdef NNUE_TRACE
        int min_bias = 1<<30, max_bias = -(1<<30);
#endif
        for (size_t i = 0; i < outputSize && s.good(); ++i) {
            _biases[i] = read_little_endian<BiasType>(s);
#ifdef NNUE_TRACE
            if (_biases[i] < min_bias) min_bias = _biases[i];
            if (_biases[i] > max_bias) max_bias = _biases[i];
#endif
        }
#ifdef NNUE_TRACE
        std::cout << "min output bias = " << min_bias << " max output bias = " << max_bias << std::endl;
#endif
        return s;
    }

    virtual const BiasType *getBiases() const noexcept { return _biases; }

    virtual const WeightType *getCol(size_t col) const noexcept { return _weights[col]; }

    virtual void setCol(size_t index, const WeightType *col) {
        for (size_t i = 0; i < inputSize; ++i)
            _weights[index][i] = col[i];
    }

    void setBiases(const BiasType *b) {
        std::memcpy(_biases,b,outputSize*sizeof(BiasType));
    }

protected:
    alignas(alignment) BiasType _biases[outputSize];
    alignas(alignment) WeightType _weights[outputSize][roundedInputSize];
};

#endif

