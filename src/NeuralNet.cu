#include <layers/affineLayer.cuh>
#include <layers/reluLayer.cuh>
#include <layers/dropoutLayer.cuh>

#include <lossFunctions/softmax.cuh>

#include <optimisers/adam.cuh>

#include <neuralNet.cuh>

#include <numC++/npGPUArray.cuh>

#include <iostream>
#include <string>
#include <vector>

NeuralNet::NeuralNet(float reg, float p_keep)
{
    this->reg = reg;
    this->relu_layers.push_back(ReLULayer());

    this->dropout_layers.push_back(DropoutLayer(p_keep));

    this->affine_layers.push_back(AffineLayer(784, 2048));
    this->affine_layers.push_back(AffineLayer(2048, 10));

    this->adam_configs.push_back(AdamOptimiser(0.001f, 0.9f, 0.999f, 1e-8f));
    this->adam_configs.push_back(AdamOptimiser(0.001f, 0.9f, 0.999f, 1e-8f));

    this->adam_configs.push_back(AdamOptimiser(0.001f, 0.9f, 0.999f, 1e-8f));
    this->adam_configs.push_back(AdamOptimiser(0.001f, 0.9f, 0.999f, 1e-8f));

    this->mode = std::string("test");
}

NeuralNet::NeuralNet(const NeuralNet &N)
{
    this->affine_layers = N.affine_layers;
    this->reg = N.reg;
    this->relu_layers = N.relu_layers;
    this->dropout_layers = N.dropout_layers;
    this->adam_configs = N.adam_configs;
    this->mode = N.mode;
}

void NeuralNet::operator=(const NeuralNet &N)
{
    this->affine_layers = N.affine_layers;
    this->reg = N.reg;
    this->relu_layers = N.relu_layers;
    this->dropout_layers = N.dropout_layers;
    this->mode = N.mode;
}

void NeuralNet::train()
{
    this->mode = std::string("train");
}
void NeuralNet::eval()
{
    this->mode = std::string("eval");
}

np::ArrayGPU<float> NeuralNet::forward(const np::ArrayGPU<float> &X)
{
    if (this->mode == "train")
    {
        std::cerr << "\nMode train but y not given";
        exit(1);
    }
    auto out = X;

    // except last_layer, all layers have activation functions and dropout
    for (int i = 0; i < this->affine_layers.size() - 1; ++i)
    {
        out = this->affine_layers[i](out, this->mode);
        out = dropout_layers[i](out);
        out = relu_layers[i](out);
    }

    // last layer no activations or dropouts
    out = affine_layers.back()(out);

    return out;
}

// return outNloss vector
std::pair<np::ArrayGPU<float>, np::ArrayGPU<float>> NeuralNet::forward(const np::ArrayGPU<float> &X, const np::ArrayGPU<int> &y)
{
    auto out = X;

    // except last_layer, all layers have activation functions and dropout
    for (int i = 0; i < this->affine_layers.size() - 1; ++i)
    {
        out = this->affine_layers[i](out, this->mode);

        out = dropout_layers[i](out, this->mode);

        out = relu_layers[i](out, this->mode);
    }

    // last layer no activations or dropouts
    out = affine_layers.back()(out, this->mode);
    // if model is in eval mode, return out and loss only. No need for backprop
    if (this->mode == "eval")
        return {out, SoftmaxLoss::computeLoss(out, y)};

    // vector of loss, dout
    auto lossNgrad = SoftmaxLoss::computeLossAndGrad(out, y);

    if (this->reg > 0)
        for (auto &al : affine_layers)
            lossNgrad.first = lossNgrad.second + (al.W * al.W).sum() * 0.5 * this->reg;

    this->backward(lossNgrad.second);

    if (this->reg > 0)
        for (auto &al : affine_layers)
            al.dW = al.dW + al.W * this->reg;

    // output vector = out, loss
    return {out, lossNgrad.first};
}

np::ArrayGPU<float> NeuralNet::operator()(const np::ArrayGPU<float> &X)
{
    return this->forward(X);
}
std::pair<np::ArrayGPU<float>, np::ArrayGPU<float>> NeuralNet::operator()(const np::ArrayGPU<float> &X, const np::ArrayGPU<int> &y)
{
    return this->forward(X, y);
}

np::ArrayGPU<float> NeuralNet::backward(np::ArrayGPU<float> &dout)
{
    // last layer backward. no relu or dropouts.
    dout = this->affine_layers.back().backward(dout);

    // relu backward -> dropout backward -> affinelayer backward
    for (int i = this->affine_layers.size() - 2; i >= 0; --i)
    {
        dout = relu_layers[i].backward(dout);
        dout = dropout_layers[i].backward(dout);

        dout = this->affine_layers[i].backward(dout);
    }

    return dout;
}

void NeuralNet::adamStep()
{
    for (int layerIdx = 0; layerIdx < this->affine_layers.size(); ++layerIdx)
    {
        // for every layer, there are 2 adam configs.
        adam_configs[layerIdx * 2].step(affine_layers[layerIdx].W, affine_layers[layerIdx].dW);
        adam_configs[layerIdx * 2 + 1].step(affine_layers[layerIdx].b, affine_layers[layerIdx].db);
    }
}
