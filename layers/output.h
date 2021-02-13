// Copyright 2021 by Jon Dart. All Rights Reserved.
#ifndef _OUTPUT_LAYER_H
#define _OUTPUT_LAYER_H

#include "layer.h"

// This class defines the output layer of the NNUE
template <typename InputType, typename OutputType, typename WeightType>
class OutputLayer : public Layer<InputType, OutputType, WeightType>
{
public:
    HiddenLayer(size_t inputSize, size_t outputSize) :
        Layer<InputType, OutputType, WeightType>(inputSize, outputSize) 
        {
        }
    
    virtual ~HiddenLayer() = default;

    const OutputType &forward(const InputType &input) {
        output = _biases + arma::dot(input,this->_weights);
    }
    
private:
    arma::Col<WeightType> _biases;
    OutputType output;
};
    
#endif
