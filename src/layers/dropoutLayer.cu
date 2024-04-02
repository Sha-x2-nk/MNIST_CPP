// layers
#include <layers/layer.cuh>
#include <layers/dropoutLayer.cuh>

// numC
#include <numC/npGPUArray.cuh>
#include <numC/npRandom.cuh>

// cuda runtime
#include <cuda_runtime.h>

// std headers
#include <iostream>

// dropout class definiton
// default and parameter constructor
DropoutLayer::DropoutLayer(const float p_keep)
{
    this->p_keep = p_keep;
}

// copy constructor
DropoutLayer::DropoutLayer(const DropoutLayer &L)
{
    this->cache = L.cache;
    this->p_keep = L.p_keep;
}

// assignment operator
DropoutLayer DropoutLayer::operator=(const DropoutLayer &L)
{
    DropoutLayer L_new;
    L_new.cache = L.cache;
    L_new.p_keep = L.p_keep;
    return L_new;
}

// ################################# forward pass #############################################
np::ArrayGPU<float> DropoutLayer::forward(const np::ArrayGPU<float> &X, const std::string &mode)
{
    /* Performs the forward pass for (inverted) dropout.

        Inputs:
        - x: Input data, of any shape
        - p_keep : Dropout parameter. We keep each neuron output with probability p.
        - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.

        Outputs:
        - out: Array of the same shape as x.

        Also stores:
        - cache: dropout mask, for backpropagation.In training mode, mask is the
        dropout mask that was used to multiply the input; in test mode, mask is not used.
    */
    if (mode == "train")
    {
        this->cache = (np::Random::rand<float>(X.rows, X.cols) < this->p_keep) / this->p_keep;
        auto out = this->cache * X;
        return out;
    }
    else
        return X;
}
// #############################################################################################

// ################################# backward pass #############################################
np::ArrayGPU<float> DropoutLayer::backward(const np::ArrayGPU<float> &dout)
{
    /* Perform the backward pass for (inverted) dropout.

        Inputs:
        - dout: Upstream derivatives, of any shape
        - cache: mask from dropout_forward.

        Outputs:
        - dX: gradients computed. same shape as dout
    */
    auto dX = dout * this->cache;
    return dX;
}
// #############################################################################################
