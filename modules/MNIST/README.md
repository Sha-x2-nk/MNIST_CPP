# MNIST Reader and Display Functions
This repository provides C++ header files for reading and displaying images from the MNIST dataset. The MNIST dataset is a widely used dataset in machine learning for handwritten digit recognition.

It also has MNIST dataset files, in original format stored in ./dataset folder.

## Contents
1. readMNIST Header
2. showMNIST Header

## readMNIST Header
The readMNIST.hpp header file contains functions for reading images and labels from MNIST files.

### Functions
1. `int reverseInt(int i)`

    Reverses the byte order of an integer.

2. `uchar* readMNISTImages(std::string &path, int &num_images, int &img_size)`

    Reads MNIST images from a file specified by path.

    * `path`: Path to the MNIST image file.
    * `num_images`: Reference to an integer to store the number of images read.
    * `img_size`: Reference to an integer to store the size of each image.
3. `uchar* readMNISTLabels(std::string &path, int &num_labels)`

    Reads MNIST labels from a file specified by path.
    * `path`: Path to the MNIST label file.
    * `num_labels`: Reference to an integer to store the number of labels read.


## showMNIST Header
The `ShowMNIST.hpp` header file contains functions for displaying MNIST images.
### Functions
1. `void showMNIST(uchar* img, int img_height, int img_width, std::string &winName)`

    Displays an MNIST image represented as unsigned char array.
    * `img`: Pointer to the image data.
    * `img_height`: Height of the image.
    * `img_width`: Width of the image.
    * `winName`: Name of the window for display.
2. `void showMNIST(float* img, int img_height, int img_width, std::string &winName)`

    Displays an MNIST image represented as float array.
    * `img`: Pointer to the image data.
    * `img_height`: Height of the image.
    * `img_width`: Width of the image.
    * `winName`: Name of the window for display.

## Usage
Include the necessary header files in your C++ project:
```cpp
#include "MNISTRead.h"
#include "ShowMNIST.h"
```
Then use the provided functions to read and display MNIST data.

## Example
```cpp
#include "MNIST/MNISTRead.h"
#include "MNIST/ShowMNIST.h"

#include <iostream>

int main() {
    std::string imagesPath = "path/to/mnist/images";
    std::string labelsPath = "path/to/mnist/labels";
    int num_images, img_size, num_labels;

    // Read MNIST images and labels
    uchar* images = readMNISTImages(imagesPath, num_images, img_size);
    uchar* labels = readMNISTLabels(labelsPath, num_labels);

    // Display an MNIST image
    showMNIST(images, 28, 28, "MNIST Image");

    // Clean up memory
    delete[] images;
    delete[] labels;

    return 0;
}
```

## Acknowledgements
The MNIST dataset used in this project is made available by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges. 
