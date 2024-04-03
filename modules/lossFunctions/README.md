# Loss Functions defined using numC
This repository contains a C++ implementation of the Softmax Loss function, which is commonly used in machine learning for classification tasks.

More loss function can/may be added in future.

## Overview
The Softmax Loss function computes both the loss and gradient for softmax classification. It takes input data x and labels y, where x is of shape (N, C) and y is of shape (N,) where N is the number of samples and C is the number of classes.

## Installation
To use this library, ensure you have the following dependencies installed:

* [numC](https://github.com/Sha-x2-nk/numC/tree/master): A C++ library for numerical computing.
* Include headers and link with softmax.cu during compilation.

## Usage
### SoftmaxLoss Class
The SoftmaxLoss class provides the following methods:
* `std::vector<np::ArrayGPU<float>> computeLossAndGrad(const np::ArrayGPU<float> &x, const np::ArrayGPU<int> &y);`

Computes the loss and gradient for softmax classification.

<b>Inputs</b>:

* `x`: Input data, of shape (N, C) where x[i, j] is the score for the jth class for the ith input.
* `y`: Vector of labels, of shape (N,) where y[i] is the label for x[i] and 0 <= y[i] < C

<b>Returns:</b>
* `loss`: Scalar giving the loss

loss: Scalar giving the loss
dx: Gradient of the loss with respect to x