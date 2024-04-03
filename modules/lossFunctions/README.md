# Loss Functions defined using numC
This repository contains a C++ implementation of the Softmax Loss function, which is commonly used in machine learning for classification tasks.

More loss functions can/may be added in future.

## Overview
The Softmax Loss function computes both the loss and gradient for softmax classification. It takes input data x and labels y, where x is of shape (N, C) and y is of shape (N,) where N is the number of samples and C is the number of classes.

## Installation
To use this library, ensure you have the following dependencies installed:

* [numC](https://github.com/Sha-x2-nk/numC/tree/master): A C++ library for numerical computing.
* Include headers and link with files in src/ during compilation.

## Usage
### SoftmaxLoss Class
The SoftmaxLoss class provides the following methods:
*  computeLossAndGrad
    ```cpp
    std::pair<np::ArrayGPU<float>, np::ArrayGPU<float>> computeLossAndGrad(const np::ArrayGPU<float> &x, const np::ArrayGPU<int> &y);
    ```

    Computes the loss and gradient for softmax classification.

    <b>Inputs</b>:

    * `x`: Input data, of shape (N, C) where x[i, j] is the score for the jth class for the ith input.
    * `y`: Vector of labels, of shape (N,) where y[i] is the label for x[i] and 0 <= y[i] < C

    <b>Returns:</b> `std::pair<np::ArrayGPU<float>, np::ArrayGPU<float>>`
    * `loss`: Scalar giving the loss
    * `dx`: Scalar giving the loss

* computeLoss 
    ```cpp
    std::vector<np::ArrayGPU<float>> computeLossAndGrad(const np::ArrayGPU<float> &x, const np::ArrayGPU<int> &y);
    ```
    Computes the loss for softmax classification.

    <b>Inputs:</b>

    * `x`: Input data, of shape (N, C) where x[i, j] is the score for the jth class for the ith input.
    * `y`: Vector of labels, of shape (N,) where y[i] is the label for x[i] and 0 <= y[i] < C
    
    <b>Returns:</b>

    * `loss`: Scalar giving the loss

### Example
```cpp
#include "lossFunctions/softmax.cuh"

#include "numC/gpuConfig.cuh"
#include "numC/npArrayGPU.cuh"
#include "numC/npFunctions.cuh"
#include "numC/npRandom.cuh"

#include <iostream>

int main() {
    // get GPU config
    np::getGPUConfig(0);

    // Sample input data and labels
    auto x = np::Random::randn<float>(2, 3); // normal distribution
    auto y = np::arange<int>(2); // array = {0, 1}

    // Compute loss and gradient
    auto loss_grad = SoftmaxLoss::computeLossAndGrad(x, y);
    std::cout << "Loss: " << loss_grad.first << std::endl;
    std::cout << "Gradient: " << loss_grad.second << std::endl;

    // Compute loss only
    auto loss = SoftmaxLoss::computeLoss(x, y);
    std::cout << "Loss: " << loss << std::endl;

    return 0;
}

```
## Acknowledgements
This project is inspired by the Softmax Loss implementation in the assignments of CS231n, a deep learning course at Stanford University.
