#include <neuralNet.cuh>

#include <MNIST/mnist_read.hpp>

#include <numC/gpuConfig.cuh>
#include <numC/npFunctions.cuh>
#include <numC/npGPUArray.cuh>

#include <visualisations/showImg.hpp>

#include <iostream>
#include <string>
#include <vector>


typedef unsigned char uchar;

// returns 2 vectors, 1 of imgs, 1 of labels
// train, val and test respectively
std::pair<std::vector<float*>, std::vector<int*>> prepareDataset(){
    int num_train_images, img_size;
    uchar* train_imgs = read_mnist_images(std::string("C:/Users/shash/Documents/jordi/dataset_mnist/train-images.idx3-ubyte"), num_train_images, img_size);

    std::cout<<"Train images fetched!. \n";
    std::cout<<"Num Images: "<<num_train_images<<" img size: "<<img_size<<std::endl; 

    int num_train_labels;
    uchar* train_labels = read_mnist_labels(std::string("C:/Users/shash/Documents/jordi/dataset_mnist/train-labels.idx1-ubyte"), num_train_labels);
    std::cout<<"Train labels fetched!. \n";
    std::cout<<"Num Labels: "<<num_train_labels<<std::endl;

    showImage(train_imgs, std::string(std::string("Ex. train img: ") + std::to_string(train_labels[0])));

    // we will shuffle indexes, to do random train val split
    auto a = np::arange<int>(num_train_images);
    np::shuffle(a);


    // out of randmly shuffled indexes, keep 2000 for validation, rest for training
    float *train_imgs_cpu = (float*)malloc(58000 * img_size * sizeof(float));
    int* train_labels_cpu = (int*)malloc(58000 * sizeof(int));
    float* val_imgs_cpu = (float*)malloc(2000 * img_size * sizeof(float));
    int* val_labels_cpu = (int*)malloc(2000 * sizeof(int));

    int in;
    std::cout << "\nWAITING FOR INP: ";
    std::cin >> in;

    for (int i = 0; i < 2000; ++i) {
        int idx = a.at(i);
        for (int img_idx = 0; img_idx < img_size; ++img_idx)
            val_imgs_cpu[idx * img_size + img_idx] = train_imgs[idx * img_size + img_idx];


        val_labels_cpu[idx] = train_labels[idx];
    }
    std::cout << "\nWAITING FOR INP: ";
    std::cin >> in;
    for (int i = 2000; i < 60000; ++i) {
        int idx = a.at(i);
        for(int img_idx = 0; img_idx < img_size; ++img_idx)
            train_imgs_cpu[idx * img_size + img_idx] = train_imgs[idx * img_size + img_idx];

        
        train_labels_cpu[idx] = train_labels[idx];
    }


    free(train_imgs);
    free(train_labels);

    
    int num_test_images;
    uchar* test_imgs = read_mnist_images(std::string("C:/Users/shash/Documents/jordi/dataset_mnist/t10k-images.idx3-ubyte"), num_test_images, img_size);

    std::cout<<"Test images fetched!. \n";
    std::cout<<"Num Images: "<<num_test_images<<" img size: "<<img_size<<std::endl; 

    int num_test_labels;
    uchar* test_labels = read_mnist_labels(std::string("C:/Users/shash/Documents/jordi/dataset_mnist/t10k-labels.idx1-ubyte"), num_test_labels);
    std::cout<<"Test labels fetched!. \n";
    std::cout<<"Num Labels: "<<num_test_labels<<std::endl;

    showImage(test_imgs, std::string(std::string("Ex. test img: ") + std::to_string(test_labels[0])));


    // out of randmly shuffled indexes, keep 2000 for validation, rest for training
    float *test_imgs_cpu = (float*)malloc(num_test_images * img_size * sizeof(float));
    int* test_labels_cpu = (int*)malloc(num_test_images * sizeof(int));

    for (int idx = 0; idx < num_test_images; ++idx) {
        for (int img_idx = 0; img_idx < img_size; ++img_idx)
            test_imgs_cpu[idx * img_size + img_idx] = train_imgs[idx * img_size + img_idx];


        test_labels_cpu[idx] = train_labels[idx];
    }

    free(test_labels);

    return {{train_imgs_cpu, val_imgs_cpu, test_imgs_cpu}, {train_labels_cpu, val_labels_cpu, test_labels_cpu}};
}

int main(){
    np::getGPUConfig(0);

    auto imgsNlabels = prepareDataset();


    cublasDestroy(np::cbls_handle);    
    return 0;
}