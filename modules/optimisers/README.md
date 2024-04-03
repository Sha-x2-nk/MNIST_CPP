# Optimisers
This repository contains a C++ implementation of the Adam Optimizer, a popular optimization algorithm used in machine learning for updating model parameters during training.

More optimisers can/may be added in future.

## Overview
The Adam Optimizer implements the Adam algorithm, which combines elements of the SGD momentum and RMSProp optimisers to efficiently update model parameters based on gradients.

## Installation
To use this library, you need to have the following dependencies installed:
* [numC](https://github.com/Sha-x2-nk/numC/tree/master): A C++ library for numerical computing.
* Include headers and link with files in src/ during compilation.

## Usage
### AdamOptimiser Class
The `AdamOptimiser` class provides the following functionality:
1. <b>Constructor</b>
    ```cpp
    AdamOptimiser(const float learning_rate = 0.001, const float beta1 = 0.9, const float beta2 = 0.999, const float epsilon = 1e-8);
    ```
    Creates an Adam optimizer object with the specified hyperparameters:

    * `learning_rate`: The learning rate for the optimizer (default is 0.001).
    * `beta1`: The beta1 parameter for the first moment update (default is 0.9).
    * `beta2`: The beta2 parameter for the second moment update (default is 0.999).
    * `epsilon`: A small value added to prevent division by zero (default is 1e-8).

2. <b>Copy Constructor</b>
    ```cpp
    AdamOptimiser(AdamOptimiser &A);
    ```
    Creates a copy of an existing Adam optimizer object.

3. <b>Assignment Operator Overload</b>
    ```cpp
    void operator=(AdamOptimiser &A);
    ```
    Performs a deep copy of an existing Adam optimizer object, if I do A = B;
4. <b>Optimizer step</b>
    ```cpp
    void step(np::ArrayGPU<float> &param, np::ArrayGPU<float> &grad);
    ```
    Updates the model parameters param using the gradients grad according to the Adam optimization algorithm.

### Example
```cpp
#include "optimisers/adam.cuh"


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
    auto dx = np::Random::rand<float>(2, 3); // from uniform distribution with values between 0 and 1

    // Sample usage of AdamOptimiser
    AdamOptimiser adam(0.001, 0.9, 0.999, 1e-8);

    adam.step(x, dx);

    // Updated params
    std::cout << "Updated Parameters: "<<x<<std::endl;

    return 0;
}
```
## Acknowledgements
This project is inspired by the Adam optimiser implementation in the assignments of CS231n, a deep learning course at Stanford University.