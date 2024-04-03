#ifndef DROPOUTLAYER_H
#define DROPOUTLAYER_H

// layers
#include <layers/layer.cuh>

// numC
#include <numC/npGPUArray.cuh>

// std headers
#include <iostream>

// inherit from Layer class
class DropoutLayer : public Layer
{
public:
    float p_keep;

    // default and parameter constructor
    DropoutLayer(const float p_keep = 1.0f);
    // copy constructor
    DropoutLayer(const DropoutLayer &L);

    // assignment operator
    void operator=(const DropoutLayer &L);

    // ################################# forward pass #############################################
    np::ArrayGPU<float> forward(const np::ArrayGPU<float> &X, const std::string &mode) override;
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
    // #############################################################################################

    // ################################# backward pass #############################################
    np::ArrayGPU<float> backward(const np::ArrayGPU<float> &dout) override;
    /* Perform the backward pass for (inverted) dropout.

        Inputs:
        - dout: Upstream derivatives, of any shape
        - cache: mask from dropout_forward.

        Outputs:
        - dX: gradients computed. same shape as dout
    */
    // #############################################################################################
};

#endif