#include <visualisations/showImg.hpp>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

typedef unsigned char uchar;

void showImage(uchar* img, std::string &winName){
    cv::Mat img_(28, 28, CV_8U);
    img_.data = img;

    cv::imshow(winName, img_);
    cv::waitKey(0);
    cv::destroyAllWindows();
}