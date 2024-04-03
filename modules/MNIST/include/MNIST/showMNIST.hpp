#ifndef SHOWMNIST_HPP
#define SHOWMNIST_HPP

#include <string>

typedef unsigned char uchar;

// display image function.
void showMNIST(uchar* img, int img_height, int img_width, std::string &winName);

// display image function overloaded to display float styled-images.
void showMNIST(float* img, int img_height, int img_width, std::string &winName);

#endif