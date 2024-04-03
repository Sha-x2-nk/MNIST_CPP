#ifndef LAYER_H
#define LAYER_H

// for numC array definitions
#include <numC/npGPUArray.cuh>

#include <iostream>

// this is the fixed layer prototype, all layers will inherit from this
class Layer
{
public:
    np::ArrayGPU<float> cache; // to store any sort of outer variable

    // ################################# forward pass ##############################################
    // virtual because our child classes will override it.
    // mode = "train" or "eval". "train" by default
    np::ArrayGPU<float> forward(const np::ArrayGPU<float> &X);
    virtual np::ArrayGPU<float> forward(const np::ArrayGPU<float> &X, const std::string &mode);
    // operator overloading to call forward function
    np::ArrayGPU<float> operator()(const np::ArrayGPU<float> &X);
    np::ArrayGPU<float> operator()(const np::ArrayGPU<float> &X, const std::string &mode);
    // #############################################################################################

    // ################################# backward pass #############################################
    // virtual because our child classes will override it
    virtual np::ArrayGPU<float> backward(const np::ArrayGPU<float> &dout);
    // #############################################################################################
};

#endif