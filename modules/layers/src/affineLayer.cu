// layers
#include <layers/affineLayer.cuh>
#include <layers/layer.cuh>

// numC
#include <numC/npGPUArray.cuh>
#include <numC/npRandom.cuh>
#include <numC/npFunctions.cuh>

// std headers
#include <cmath>

// definition for AffineLayer class

// default and parameterised constructor
AffineLayer::AffineLayer(int in_features, int out_features)
{
    this->W = np::Random::randn<float>(in_features, out_features) * sqrtf(2.0 / in_features);
    this->b = np::zeros<float>(1, out_features);
}

// copy constructor
AffineLayer::AffineLayer(const AffineLayer &L)
{
    this->W = L.W;
    this->b = L.b;
    this->dW = L.dW;
    this->db = L.db;
    this->cache = L.cache;
}

// assignment operator
void AffineLayer::operator=(const AffineLayer &L)
{
    this->W = L.W;
    this->b = L.b;
    this->dW = L.dW;
    this->db = L.db;
    this->cache = L.cache;
};

// ################################# forward pass ##############################################
np::ArrayGPU<float> AffineLayer::forward(const np::ArrayGPU<float> &X, const std::string &mode)
{
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
    np::ArrayGPU<float> out = X.dot(this->W) + this->b;

    if (mode == "train")
        this->cache = X;

    return out;
}
// #############################################################################################

// ################################# backward pass #############################################
np::ArrayGPU<float> AffineLayer::backward(const np::ArrayGPU<float> &dout)
{
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
    this->db = dout.sum(0);
    this->dW = this->cache.Tdot(dout);
    auto dX = dout.dotT(this->W);

    return dX;
}
// #############################################################################################
