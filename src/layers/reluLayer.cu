// layers
#include <layers/layer.cuh>
#include <layers/reluLayer.cuh>

// numC
#include <numC/npGPUArray.cuh>
#include <numC/npFunctions.cuh>

// cuda runtime
#include <cuda_runtime.h>

// std headers
#include <iostream>

// relu layer definiton

// default constructor
ReLULayer::ReLULayer()
{
    ;
}

// copy constructor
ReLULayer::ReLULayer(const ReLULayer &L)
{
    this->cache = L.cache;
}

// assignment operator
void ReLULayer::operator=(const ReLULayer &L)
{
    this->cache = L.cache;
}

// ################################# forward pass ##############################################
np::ArrayGPU<float> ReLULayer::forward(const np::ArrayGPU<float> &X, const std::string &mode)
{
    /* Computes the forward pass for a layer of rectified linear units (ReLUs).

        Input:
        - X: Inputs, of any shape

        Returns:
        - out: Output, of the same shape as x

        Also stores:
        - cache: x, for backpropagation
    */
    auto out = np::maximum<float>(X, 0);
    if (mode == "train")
        this->cache = X;
    return out;
}
// #############################################################################################

// ################################# backward pass #############################################
np::ArrayGPU<float> ReLULayer::backward(const np::ArrayGPU<float> &dout)
{
    /* Computes the backward pass for a layer of rectified linear units (ReLUs).

        Input:
        - dout: Upstream derivatives, of any shape
        - cache: Input x, of same shape as dout

        Returns:
        - dx: Gradient with respect to x
    */
    auto dX = dout * (this->cache > 0);
    return dX;
    // #############################################################################################
}