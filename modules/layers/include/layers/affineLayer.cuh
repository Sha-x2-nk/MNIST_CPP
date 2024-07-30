#ifndef AFFINELAYER_CUH
#define AFFINELAYER_CUH

// layers
#include <layers/layer.cuh>

// numC++
#include <numC++/npGPUArray.cuh>

// inherit from Layer class
class AffineLayer : public Layer
{
public:
    // weights and bias
    np::ArrayGPU<float> W, b;

    // gradients of weights and bias
    np::ArrayGPU<float> dW, db;

    // default and parameter constructor
    AffineLayer(int in_features = 1, int out_features = 1);
    // copy constructor
    AffineLayer(const AffineLayer &L);

    // assignment operator
    void operator=(const AffineLayer &L);

    // ################################# forward pass ##############################################
    np::ArrayGPU<float> forward(const np::ArrayGPU<float> &X, const std::string &mode) override;
    /* Computes the forward pass for an affine (fully connected) layer.

        The input x has shape (N, D) and contains a minibatch of N
        examples, where each example x[i] has shape D. We will
        transform it to an output vector of dimension M.

        Inputs:
        - x: A numpy array containing input data, of shape (N, D)
        - w: A numpy array of weights, of shape (D, M)
        - b: A numpy array of biases, of shape (M,)

        Returns:
        - out: output, of shape (N, M)

        Also stores:
        - cache: x, for backpropagation
    */
    // #############################################################################################

    // ################################# backward pass #############################################
    np::ArrayGPU<float> backward(const np::ArrayGPU<float> &dout) override;
    /* Computes the backward pass for an affine (fully connected) layer.

        Inputs:
        - dout: Upstream derivative, of shape (N, M)
        - cache: Input data X, of shape (N, D)
        - W: Weights, of shape (D, M)
        - b: Biases, of shape (M,)

        Returns:
        - dX: Gradient with respect to x, of shape (N, D)

        Also computes and stores:
        - dW: Gradient with respect to w, of shape (D, M)
        - db: Gradient with respect to b, of shape (M,)
    */
    // #############################################################################################
};

#endif