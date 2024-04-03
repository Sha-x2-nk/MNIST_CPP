// layers
#include <layers/layer.cuh>

// numC files
#include <numC/npGPUArray.cuh>

// cuda runtime
#include <cuda_runtime.h>

// std headers
#include <iostream>

// ################################# forward pass ##############################################
// virtual because our child classes will override it.
// mode = "train" or "eval". "train" by default
np::ArrayGPU<float> Layer::forward(const np::ArrayGPU<float> &X)
{
    return this->forward(X, std::string("train"));
}
np::ArrayGPU<float> Layer::forward(const np::ArrayGPU<float> &X, const std::string &mode)
{
    std::cout << "Implement this function in child class";
    return np::ArrayGPU<float>(1, 1);
}
// operator overloading to call forward function
np::ArrayGPU<float> Layer::operator()(const np::ArrayGPU<float> &X)
{
    return this->forward(X);
}
np::ArrayGPU<float> Layer::operator()(const np::ArrayGPU<float> &X, const std::string &mode)
{
    return this->forward(X, mode);
}
// #############################################################################################

// ################################# backward pass #############################################
// virtual because our child classes will override it
np::ArrayGPU<float> Layer::backward(const np::ArrayGPU<float> &dout)
{
    std::cout << "Implement this function in child class";
    return np::ArrayGPU<float>(1, 1);
}
// #############################################################################################
