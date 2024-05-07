// Copyright 2022, 2024 by Jon Dart. All Rights Reserved
#ifndef _NNUE_SQRCRELU_H
#define _NNUE_SQRCRELU_H

#include "typed.h"

// This combines both a SqrCReLU operation and a linear layer with output size one, the (single) hidden layer
// in the Arasan V3 architecture.
template <typename InputType, typename AccumulatorType, typename WeightType, typename BiasType, typename OutputType, size_t inputSize,
          unsigned clampMax, size_t alignment = DEFAULT_ALIGN>
class SqrCReLUAndLinear
    : public LinearLayer<InputType, WeightType, BiasType, OutputType, inputSize, 1, alignment> {
  public:
    SqrCReLUAndLinear() = default;

    virtual ~SqrCReLUAndLinear() = default;

    virtual void doForward([[maybe_unused]] const InputType *input, [[maybe_unused]] OutputType *output) const noexcept {
        // no-op for this layer: use method below
        assert(0);
    }

    void postProcessAccum(const AccumulatorType &accum, OutputType *output) const {
        //#if defined(SIMD)
        //        if constexpr (sizeof(InputType) == 2 && sizeof(OutputType) == 1) {
        //            simd::sqrCRelUAndLinear<InputType, OutputType, size / 2, clampMax, scaleFactor>(
        //                accum.getOutput(AccumulatorHalf::Lower), output);
        //            simd::sqrCRelUAndLinear<InputType, OutputType, size / 2, clampMax, scaleFactor>(
        //                accum.getOutput(AccumulatorHalf::Upper), output + size / 2);
        //        } else
        //#endif
        {
            // generic implementation
            int sum = 0;
            for (size_t i = 0; i < accum.getSize(); ++i) {
                InputType x = accum.getOutput(AccumulatorHalf::Lower)[i];
                InputType y = accum.getOutput(AccumulatorHalf::Upper)[i];
                // CReLU
                x = std::clamp<int>(x, 0, clampMax);
                y = std::clamp<int>(y, 0, clampMax);
                // pairwise product of the accumulator halves
                int product = x * y;
                // vector multiply and sum
                sum += product * this->_weights[0][i];
            }
            output[0] = (sum / NETWORK_QA) + this->_biases[0];
        }
#ifdef NNUE_TRACE
            std::cout << "---- SqrCReLUAndLinear output " << std::endl;
        for (size_t i = 0; i < 1 /*outputSize */; ++i) {
            std::cout << static_cast<int>(output[i]) << ' ';
            if ((i + 1) % 64 == 0)
                std::cout << std::endl;
        }
        std::cout << std::endl;
#endif
    }

};

#endif
