#ifndef RELULAYER_H
#define RELULAYER_H

// layers
#include <layers/layer.cuh>

// numC
#include <numC/npGPUArray.cuh>

// std headers
#include <iostream>

// inherit from Layer class
class ReLULayer : public Layer
{
public:
    // default constructor
    ReLULayer();
    // copy constructor
    ReLULayer(const ReLULayer &r);

    // assignment operator
    ReLULayer operator=(const ReLULayer &L);

    // ################################# forward pass ##############################################
    np::ArrayGPU<float> forward(const np::ArrayGPU<float> &X, const std::string &mode) override;
    /* Computes the forward pass for a layer of rectified linear units (ReLUs).

        Input:
        - X: Inputs, of any shape

        Returns:
        - out: Output, of the same shape as x

        Also stores:
        - cache: x, for backpropagation
    */
    // #############################################################################################

    // ################################# backward pass #############################################
    np::ArrayGPU<float> backward(const np::ArrayGPU<float> &dout) override;
    /* Computes the backward pass for a layer of rectified linear units (ReLUs).

        Input:
        - dout: Upstream derivatives, of any shape
        - cache: Input x, of same shape as dout

        Returns:
        - dx: Gradient with respect to x
    */
    // #############################################################################################
};

#endif