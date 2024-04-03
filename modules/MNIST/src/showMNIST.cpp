#include <MNIST/showMNIST.hpp>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

typedef unsigned char uchar;

// display MNIST images
void showMNIST(uchar* img, int img_height, int img_width, std::string &winName){
    cv::Mat img_(img_height, img_width, CV_8U);
    img_.data = img;

    cv::imshow(winName, img_);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

// display MNIST images, overloaded function to display float converted images
void showMNIST(float* img, int img_height, int img_width, std::string &winName){
    uchar *img_ = new uchar[img_height * img_width];

    for(int i= 0; i< img_height * img_width; ++i)
        img_[i] = img[i];

    showMNIST(img_, img_height, img_width, winName);
}