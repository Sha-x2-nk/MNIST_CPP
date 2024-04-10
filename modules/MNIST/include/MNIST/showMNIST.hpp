#ifndef SHOWMNIST_HPP
#define SHOWMNIST_HPP

#include <string>

typedef unsigned char uchar;

// display image function.
void showMNIST(const uchar *img, const int img_height, const int img_width, const std::string &winName);

// display image function overloaded to display float styled-images.
void showMNIST(const float *img, const int img_height, const int img_width, const std::string &winName);

#endif