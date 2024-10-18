// Copyright 2024 by Jon Dart. All Rights Reserved
#ifndef _NNUE_SQRCRELUANDLINEAR_H
#define _NNUE_SQRCRELUANDLINEAR_H

#include "typed.h"

// This combines both a SqrCReLU operation and a linear layer with output size one, the (single) hidden layer
// in the Arasan V3 architecture. TBD: generalize to larger output sizes.
// If saturate = true, the weights are multiplied by the input and the high 16 bits discarded. Then the operation
// is completed by multiplying by the input again (squaring)
// If saturate = false, the input is just squared, asssuming no saturation. This will be true if the inputs are
// clamped to <=181 = square root of 32768.
template <typename AccumulatorType, typename InputType, typename WeightType, typename BiasType, typename OutputType, size_t inputSize,
          unsigned clampMax, size_t NETWORK_QA, size_t buckets, bool saturate, size_t alignment = DEFAULT_ALIGN>
class SqrCReLUAndLinear
    : public LinearLayer<InputType, WeightType, BiasType, OutputType, inputSize, 1, buckets, alignment> {
  public:
    SqrCReLUAndLinear() = default;

    virtual ~SqrCReLUAndLinear() = default;

    virtual void forward([[maybe_unused]] size_t bucket, [[maybe_unused]] const InputType *input,
                         [[maybe_unused]] OutputType *output) const noexcept {
        // no-op for this layer: use method below
        assert(0);
    }

    void postProcessAccum(const AccumulatorType &accum, unsigned bucket, OutputType *output) const {
        int32_t sum = 0;
#ifdef SIMD
        if constexpr (sizeof(InputType) == 2) {
            // SIMD optimized implementation
            simd::sqrCRelUAndLinear < InputType, OutputType, WeightType, inputSize / 2, 1, saturate >
                                      (accum.getOutput(AccumulatorHalf::Lower), output, clampMax,
                                       this->_weights[bucket][0]);
            sum += *output;
            simd::sqrCRelUAndLinear < InputType, OutputType, WeightType, inputSize / 2, 1, saturate >
                                      (accum.getOutput(AccumulatorHalf::Upper), output, clampMax,
                                       this->_weights[bucket][0] + (inputSize / 2));
            sum += *output;
            output[0] = (sum / NETWORK_QA) + this->_biases[bucket][0];
        } else
#endif
        {
            // generic implementation
            static AccumulatorHalf halves[] = {AccumulatorHalf::Lower, AccumulatorHalf::Upper};
            size_t offset = 0;
            for (auto h : halves) {
                for (size_t i = 0; i < accum.getSize(); ++i) {
                    int16_t x = accum.getOutput(h)[i];
                    // CReLU
                    x = std::clamp<int16_t>(x, 0, clampMax);
                    if constexpr (saturate) {
                        sum += std::clamp<int32_t>(this->_weights[bucket][0][i + offset] * x,-32767,32768) * x;
                    }
                    else {
                        // square and sum
                        sum += (this->_weights[bucket][0][i + offset] * x * x);
                    }
                }
                offset += accum.getSize();
            }
            // convert sum to a range that corrects for the squaring, i.e.
            // what it would have if this were a regular CReLU layer
            output[0] = (sum / NETWORK_QA) + this->_biases[bucket][0];
        }
#ifdef NNUE_TRACE
        std::cout << "---- SqrCReLUAndLinear output " << std::endl;
        std::cout << " prescaled = " << sum << " unsquared = " << output[0] << std::endl;
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
