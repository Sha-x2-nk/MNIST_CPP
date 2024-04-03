#ifndef showMNIST_H
#define showMNIST_H

#include <iostream>
#include <string>

typedef unsigned char uchar;

// display image function.
void showMNIST(uchar* img, int img_height, int img_width, std::string &winName);

// display image function overloaded to display float styled-images.
void showMNIST(float* img, int img_height, int img_width, std::string &winName);

#endif