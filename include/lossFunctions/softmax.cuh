#ifndef SOFTMAX_H
#define SOFTMAX_H

// numC
#include <numC/npGPUArray.cuh>

// std
#include <iostream>
#include <vector>

class SoftmaxLoss
{
public:
    std::vector<np::ArrayGPU<float>> computeLossAndGrad(const np::ArrayGPU<float> &x, const np::ArrayGPU<int> &y);
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

    np::ArrayGPU<float> computeLoss(const np::ArrayGPU<float> &x, const np::ArrayGPU<int> &y);
    /*Computes the loss and gradient for softmax classification.

         Inputs:
         - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
         class for the ith input.
         - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
         0 <= y[i] < C

         Returns:
         - loss: Scalar giving the loss
     */
};

#endif