#include <MNIST/readMNIST.hpp>

#include <iostream>
#include <fstream>
#include <string>

typedef unsigned char uchar;

int reverseInt(const int i)
{
    unsigned char c1, c2, c3, c4;
    c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

// reading mnist images from file.
uchar *readMNISTImages(const std::string &path, int &num_images, int &img_size)
{
    std::ifstream file(path, std::ios::binary);

    if (file.is_open())
    {
        int magic_number = 0, n_rows = 0, n_cols = 0;

        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if (magic_number != 2051)
        {
            std::cerr << "Invalid MNIST image file!\n";
            return NULL;
        }

        file.read((char *)&num_images, sizeof(num_images));
        num_images = reverseInt(num_images);

        file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
        file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

        img_size = n_rows * n_cols;

        uchar *_dataset = new uchar[num_images * img_size];

        file.read(reinterpret_cast<char *>(_dataset), num_images * img_size);

        return _dataset;
    }
    else
    {
        std::cerr << "Cannot open file `" << path << "`!\n";
        return NULL;
    }
}

// reading mnist labels from file.
uchar *readMNISTLabels(const std::string &path, int &num_labels)
{

    std::ifstream file(path, std::ios::binary);

    if (file.is_open())
    {
        int magic_number = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if (magic_number != 2049)
        {
            std::cerr << "Invalid MNIST label file!\n";
            return NULL;
        }

        file.read((char *)&num_labels, sizeof(num_labels));
        num_labels = reverseInt(num_labels);

        uchar *_dataset = new uchar[num_labels];
        for (int i = 0; i < num_labels; i++)
        {
            file.read((char *)&_dataset[i], 1);
        }
        return _dataset;
    }
    else
    {
        std::cerr << "Unable to open file `" << path << "`!\n";
        return NULL;
    }
}