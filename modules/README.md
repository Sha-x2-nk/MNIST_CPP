# Modules

This folder contains various C++ header files for implementing different modules and functionalities related to machine learning and deep learning tasks.

## Contents

1. [Layers Project](#layers-project)
2. [numC++ Library](#numC++-library)
3. [Loss Functions](#loss-functions)
4. [MNIST](#mnist)
5. [Optimisers](#optimisers)

## Layers Project

The "layers" project within this folder consists of C++ header files that define neural network layers such as fully connected layers, dropout layers, and ReLU activation layers. Each header file provides implementations and functionalities for specific layer types.

- `layer.cuh`: Base class for neural network layers, with virtual functions for forward and backward passes.
- `affineLayer.cuh`: Implementation of fully connected (affine) layers inheriting from the base layer class.
- `dropoutLayer.cuh`: Implementation of dropout layers for regularization purposes.
- `reluLayer.cuh`: Implementation of the ReLU activation layer for introducing non-linearity.

For more details about the layers project, refer to the [Layers Project README](./layers/README.md).

## numC++ Library

The "numC++" library provides C++ header files for numerical computing, including array operations, linear algebra functions, and GPU acceleration support, all while maintaining numPy-like syntax.

- `npGPUArray.cuh`: Header file for GPU-accelerated numerical arrays.
- Other headers for matrix operations, element-wise operations, and more.

For more details about the numC++ library, refer to the [numC++ Library README](./numC++/README.md).

## Loss Functions

The "lossFunctions" module contains C++ header files for various loss functions commonly used in machine learning models.

- `softmaxLoss.cuh`: Implementation of the softmax loss function for classification tasks.

For more details about the loss functions module, refer to the [Loss Functions README](./lossFunctions/README.md).

## MNIST

The "MNIST" module provides functionalities for reading, preprocessing, and working with the MNIST dataset, which is widely used for handwritten digit recognition tasks.

- `mnistRead.cuh`: Header file for reading MNIST images and labels.
- `showMNIST.cuh`: Header file for displaying MNIST images.

For more details about the MNIST module, refer to the [MNIST README](./MNIST/README.md).

## Optimisers

The "optimisers" module includes C++ header files for optimization algorithms commonly used in training neural networks.

- `adamOptimiser.cuh`: Implementation of the Adam optimizer.

For more details about the optimisers module, refer to the [Optimisers README](./optimisers/README.md).

## Usage

To use the modules and functionalities provided in this folder, include the necessary header files in your C++ projects. Ensure that you follow the specific usage instructions and guidelines mentioned in each module's README file.

## Acknowledgements
Acknowledgements for each project have been specified in their respective readme.

---

**Note**: The above information is a general overview of the "modules" folder. For detailed information about specific modules and their functionalities, refer to the respective README files within each module.
