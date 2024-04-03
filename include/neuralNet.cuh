#ifndef NEURALNET_H
#define NEURALNET_H

// layers
#include <layers/affineLayer.cuh>
#include <layers/reluLayer.cuh>
#include <layers/dropoutLayer.cuh>

// loss functions
#include <lossFunctions/softmax.cuh>

// optimisers
#include <optimisers/adam.cuh>

// numC
#include <numC/npGPUArray.cuh>

// std headers
#include <iostream>
#include <string>
#include <vector>

class NeuralNet
{
public:
    // vector of affine layers. (since there can be multiple layers in neural net)
    std::vector<AffineLayer> affine_layers;
    // vector of affine layers.
    std::vector<ReLULayer> relu_layers;
    // vector of affine layers.
    std::vector<DropoutLayer> dropout_layers;

    // softmax loss.
    SoftmaxLoss softmax;

    // vector of adam configs. every parameter will have their own adam config. (m, v arrays)
    std::vector<AdamOptimiser> adam_configs;

    // regularisation strength
    float reg;

    // mode = [eval | train]
    std::string mode;

    // default + parameter constructor
    NeuralNet(const float reg = 0.0, float p_keep = 1.0);

    // copy constructor
    NeuralNet(const NeuralNet &N);

    // assignment operator overload
    void operator=(const NeuralNet &N);

    // functions to switch modes
    void train();
    void eval();

    // forward pass of model
    // this is when only x is passed. used in eval mode.
    // returns only output array after forward pass
    np::ArrayGPU<float> forward(const np::ArrayGPU<float> &X);

    // used in train mode. x and y (labels) are given.
    // return a pair of arrays, first one is out, second is loss
    std::pair<np::ArrayGPU<float>, np::ArrayGPU<float>> forward(const np::ArrayGPU<float> &X, const np::ArrayGPU<int> &y);

    // overloaded operators to perform forward pass.
    np::ArrayGPU<float> operator()(const np::ArrayGPU<float> &X);
    std::pair<np::ArrayGPU<float>, np::ArrayGPU<float>> operator()(const np::ArrayGPU<float> &X, const np::ArrayGPU<int> &y);

    // backward pass of neural network
    np::ArrayGPU<float> backward(np::ArrayGPU<float> &dout);

    // take optimiser step. (update parameters)
    void adamStep();
};

#endif