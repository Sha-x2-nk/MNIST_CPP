#ifndef NEURALNET_H
#define NEURALNET_H

#include <layers/affineLayer.cuh>
#include <layers/reluLayer.cuh>
#include <layers/dropoutLayer.cuh>

#include <lossFunctions/softmax.cuh>

#include <optimisers/adam.cuh>

#include <numC/npGPUArray.cuh>

#include <iostream>
#include <string>
#include <vector>

class NeuralNet
{
public:
    std::vector<AffineLayer> affine_layers;
    std::vector<ReLULayer> relu_layers;
    std::vector<DropoutLayer> dropout_layers;

    SoftmaxLoss softmax;

    std::vector<AdamOptimiser> adam_configs;

    float reg;
    std::string mode;

    NeuralNet(const float reg = 0.0, float p_keep = 1.0);

    NeuralNet(const NeuralNet &N);

    NeuralNet operator=(const NeuralNet &N);

    void train();
    void test();

    np::ArrayGPU<float> forward(const np::ArrayGPU<float> &X);
    std::vector<np::ArrayGPU<float>> forward(const np::ArrayGPU<float> &X, const np::ArrayGPU<int> &y);

    np::ArrayGPU<float> operator()(const np::ArrayGPU<float> &X);
    std::vector<np::ArrayGPU<float>> operator()(const np::ArrayGPU<float> &X, const np::ArrayGPU<int> &y);

    np::ArrayGPU<float> backward(np::ArrayGPU<float> &dout);

    void adamStep();
};

#endif