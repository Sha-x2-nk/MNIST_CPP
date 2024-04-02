#ifndef MNISTREAD_H
#define MNISTREAD_H

#include <iostream>
#include <string>

typedef unsigned char uchar;

int reverseInt(int i);

uchar* read_mnist_images(std::string &path, int &num_images, int &img_size);

uchar* read_mnist_labels(std::string &path, int &num_labels);

#endif