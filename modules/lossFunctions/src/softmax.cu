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

    int sz = x.rows();
    int NUM_CLASSES = x.cols();

    // scores = scores + 1e-8; // epsilon to prevent -log(0)
    // auto loss = (-np::log(scores.get(np::arange<int>(x.rows()), y))).sum() / x.rows();
    // scores.set(np::arange<int>(x.rows()), y, NP_OP_SUB, 1);


    np::ArrayGPU<float> scores_for_loss(sz);

    int BLOCK_SIZE = np::GPU_NUM_CUDA_CORE;
    dim3 block(BLOCK_SIZE);
    dim3 grid(std::min<int>(ceil(static_cast<float>(sz)/block.x), 2 * np::GPU_NUM_SM));

    kernelSoftMaxUtils<float><<<grid, block>>>(scores.mat, y.mat, scores_for_loss.mat, NUM_CLASSES, sz);
    cudaDeviceSynchronize();
    auto loss = scores_for_loss.sum()/sz;

    scores = scores / sz;

    return {loss, scores};
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

    auto loss = (-np::log(scores.get(np::arange<int>(x.rows()), y))).sum() / x.rows();

    return loss;
}