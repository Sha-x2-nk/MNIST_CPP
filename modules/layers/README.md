# Layers for Neural Nets
This repository contains a C++ header file implementation for various neural network layers. These layers are designed to be used in deep learning models for tasks such as image recognition, natural language processing, and more.

## Layers Overview
The layers included in this project are implemented using the inheritance concept, where all layers inherit from a base `Layer` class. Each layer overrides virtual functions for the forward and backward passes defined in base class, allowing for flexibility and customization in neural network architectures.

### Layer types
1. `Layer (Base Class)`: Defines the base layer with virtual functions for forward and backward passes. Defines () operator to call forward function.
2. `Affine Layer`: Represents a fully connected layer with weights and biases.
3. `Dropout Layer`: Implements dropout regularization to prevent overfitting during training.
4. `ReLU Layer`: Implements rectified linear unit activation function for non-linearity.

## File Structure
* `layer.cuh`: Base class `Layer` definition with virtual functions for forward and backward passes.
* `affineLayer.cuh`: Implementation of the fully connected (affine) layer inheriting from the Layer class.
* `dropoutLayer.cuh`: Implementation of the dropout layer inheriting from the Layer class.
* `reluLayer.cuh`: Implementation of the ReLU activation layer inheriting from the Layer class.

## Usage
Include the necessary header files in your C++ project to use these layers:

```cpp
#include <layers/layer.cuh>
#include <layers/affineLayer.cuh>
#include <layers/dropoutLayer.cuh>
#include <layers/reluLayer.cuh>
```
Create instances of the desired layers and use their forward and backward functions in your neural network models.

## Example
```cpp
#include <layers/reluLayer.cuh>

#include <iostream>

int main() {
    // Create an instance of the ReLU layer
    ReLULayer reluLayer;

    // Create an input array
    np::ArrayGPU<float> input({{1, -2, 3}, {-4, 5, -6}});

    // Forward pass through the ReLU layer
    auto output = reluLayer.forward(input, "train");

    // Display the output
    std::cout << "ReLU Layer Output:" << std::endl;
    std::cout << output << std::endl;

    // Perform backward pass through the ReLU layer
    auto gradient = reluLayer.backward(output);

    // Display the gradient
    std::cout << "ReLU Layer Gradient:" << std::endl;
    std::cout << gradient << std::endl;

    return 0;
}
```
## Acknowledgements
This project draws inspiration from deep learning frameworks and concepts taught in various courses, including CS231n (Convolutional Neural Networks for Visual Recognition) by Stanford University.