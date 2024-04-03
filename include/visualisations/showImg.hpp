#ifndef SHOWIMG_H
#define SHOWIMG_H

#include <iostream>
#include <string>

typedef unsigned char uchar;

void showImage(uchar* img, int img_height, int img_width, std::string &winName);
void showImage(float* img, int img_height, int img_width, std::string &winName);


#endif