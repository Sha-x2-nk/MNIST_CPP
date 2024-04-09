// numC
#include <numC/npGPUArray.cuh>
#include <numC/npFunctions.cuh>

// loss function
#include <lossFunctions/softmax.cuh>

// std
#include <vector>

std::pair<np::ArrayGPU<float>, np::ArrayGPU<float>> SoftmaxLoss::computeLossAndGrad(const np::ArrayGPU<float> &x, const np::ArrayGPU<int> &y)
{
    /*Computes the loss and gradient for softmax classification.

        Inputs:
        - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
        class for the ith input.
        - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
        0 <= y[i] < C

        Returns a vector of:
        - loss: Scalar giving the loss
        - dx: Gradient of the loss with respect to x
    */
    auto exp_x = np::exp(x - x.max(1));
    auto scores = exp_x / exp_x.sum(1);

    scores = scores + 1e-8; // epsilon to prevent -log(0)

    auto loss = (-np::log(scores.at(np::arange<int>(x.rows), y))).sum() / x.rows;

    auto dx = scores;
    dx.set(np::arange<int>(x.rows), y, dx.at(np::arange<int>(x.rows), y) - 1);

    for(int i= 0; i< n; ++i){
        dx.at(i, y[i]) -= 1;
    }
    dx = dx / x.rows;
    return {loss, dx};
}

np::ArrayGPU<float> SoftmaxLoss::computeLoss(const np::ArrayGPU<float> &x, const np::ArrayGPU<int> &y)
{
    /*Computes the loss and gradient for softmax classification.

        Inputs:
        - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
        class for the ith input.
        - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
        0 <= y[i] < C

        Returns:
        - loss: Scalar giving the loss
    */
    auto exp_x = np::exp(x - x.max(1));
    auto scores = exp_x / exp_x.sum(1);

    scores = scores + 1e-8; // epsilon to prevent -log(0)

    auto loss = (-np::log(scores.at(np::arange<int>(x.rows), y))).sum() / x.rows;

    return loss;
}