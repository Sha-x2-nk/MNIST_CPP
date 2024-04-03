#include <neuralNet.cuh>

#include <MNIST/readMNIST.hpp>
#include <MNIST/showMNIST.hpp>

#include <numC/gpuConfig.cuh>
#include <numC/npFunctions.cuh>
#include <numC/npGPUArray.cuh>
#include <numC/npRandom.cuh>


#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <random>

// utility function to calculate ceil(a/b) where a, b are ints
#define ceil(a, b) (a + b - 1)/b

typedef unsigned char uchar;

// returns 2 vectors, 1 of imgs, 1 of labels
// train, val and test respectively
std::pair<std::vector<float*>, std::vector<int*>> prepareDataset();

// function to train neural net
NeuralNet trainModel(float *x_train, int *y_train, int train_size, float *x_val, int *y_val, int val_size, float *x_test, int *y_test, int test_size, int img_size);

int main(){
    np::getGPUConfig(0);
    std::cout<<std::endl<<"----------------------------------------------------"<<std::endl;
    std::cout<<"----------------STARTING DATA FETCH-----------------"<<std::endl;
    std::cout<<"----------------------------------------------------"<<std::endl;

    auto imgsNlabels = prepareDataset();

    std::cout<<std::endl<<"----------------------------------------------------"<<std::endl;
    std::cout<<"-------------IMAGES AND LABELS FETCHED--------------"<<std::endl;
    std::cout<<"----------------------------------------------------"<<std::endl;

    std::cout<<std::endl<<"----------------------------------------------------"<<std::endl;
    std::cout<<"-------------Beginning GPU execution----------------"<<std::endl;
    std::cout<<"----------------------------------------------------"<<std::endl;
    
    NeuralNet nn = trainModel(imgsNlabels.first[0], imgsNlabels.second[0], 58000, imgsNlabels.first[1], imgsNlabels.second[1], 2000, imgsNlabels.first[2], imgsNlabels.second[2], 10000, 784);
    

    // viewing columns where test data was misclassified
    nn.eval();
    std::vector<int> miss_classified_test_idxs;
    for(int i= 0; i< 10000; ++i){
        auto x = np::ArrayGPU<float>(1, 784);
        x.copyFromCPU(imgsNlabels.first[2] + i * 784);

        auto y_label = (imgsNlabels.second[2] + i)[0];
        auto y_pred = nn(x).argmax(1).at(0);

        if(y_pred != y_label){
            showMNIST(imgsNlabels.first[2] + i * 784, 28, 28, std::string(std::string("Test Img. actual: ") + std::to_string(y_label) + std::string(" predicted: ") + std::to_string(y_pred)));
        }
        

    }
    
    // releasing memory occupied 
    for(int i= 0; i< 3; ++i){
        free(imgsNlabels.first[i]);
        free(imgsNlabels.second[1]);
    }
    cublasDestroy(np::cbls_handle);    
    return 0;
}

// returns 2 vectors, 1 of imgs, 1 of labels
// train, val and test respectively
std::pair<std::vector<float*>, std::vector<int*>> prepareDataset(){
    
    int num_train_images, img_size;
    uchar* train_imgs = readMNISTImages(std::string("C:/Users/shash/Documents/MNIST_NUMC/modules/MNIST/dataset/train-images.idx3-ubyte"), num_train_images, img_size);

    std::cout<<std::endl<<"[+] Train images fetched!"<<std::endl;
    std::cout<<"----------Num Images: "<<num_train_images<<" img size: "<<img_size<<"-----------"<<std::endl; 

    int num_train_labels;
    uchar* train_labels = readMNISTLabels(std::string("C:/Users/shash/Documents/MNIST_NUMC/modules/MNIST/dataset/train-labels.idx1-ubyte"), num_train_labels);
    std::cout<<std::endl<<"[+] Train labels fetched!"<<std::endl;
    std::cout<<"-----------Num Labels: "<<num_train_labels<<"------------------------"<<std::endl;

    // displaying random image out of train set.
    int random_idx = (rand() % num_train_images); 
    // since it is a 2d array, need to skip 784 elements per row.
    showMNIST(train_imgs + random_idx * 784, 28, 28, std::string(std::string("Ex. train img: ") + std::to_string(train_labels[random_idx])));

    std::cout<<std::endl<<"[.] Starting conversion to float and random train-val split"<<std::endl;

    // we will shuffle indexes, to do random train val split
    auto randIdxs = np::arange<int>(num_train_images);
    np::shuffle(randIdxs);

    const int num_val_images = 2000;

    // out of randmly shuffled indexes, keep 2000 for validation, rest for training
    float *train_imgs_cpu = (float*)malloc((num_train_images - num_val_images) * img_size * sizeof(float));
    int* train_labels_cpu = (int*)malloc((num_train_images - num_val_images) * sizeof(int));
    float* val_imgs_cpu = (float*)malloc(num_val_images * img_size * sizeof(float));
    int* val_labels_cpu = (int*)malloc(num_val_images * sizeof(int));

    for (int i = 0; i < num_val_images; ++i) {
        int idx = randIdxs.at(i);

        for (int img_idx = 0; img_idx < img_size; ++img_idx)
            val_imgs_cpu[i * img_size + img_idx] = train_imgs[idx * img_size + img_idx];


        val_labels_cpu[i] = train_labels[idx];
    }

    for (int i = num_val_images; i < num_train_images; ++i) {
        int idx = randIdxs.at(i);

        for(int img_idx = 0; img_idx < img_size; ++img_idx)
            train_imgs_cpu[(i - num_val_images) * img_size + img_idx] = train_imgs[idx * img_size + img_idx];

        
        train_labels_cpu[(i - num_val_images)] = train_labels[idx];
    }


    free(train_imgs);
    free(train_labels);

    std::cout<<"[+] Done. "<<std::endl;
    std::cout<<"------Validation set size: "<<num_val_images<<" Train set size: "<<num_train_images<<"-----"<<std::endl;
    
    int num_test_images;
    uchar* test_imgs = readMNISTImages(std::string("C:/Users/shash/Documents/MNIST_NUMC/modules/MNIST/dataset/t10k-images.idx3-ubyte"), num_test_images, img_size);

    std::cout<<std::endl<<"[+] Test images fetched!"<<std::endl;
    std::cout<<"-----------Num Images: "<<num_test_images<<" img size: "<<img_size<<"------------"<<std::endl; 

    int num_test_labels;
    uchar* test_labels = readMNISTLabels(std::string("C:/Users/shash/Documents/MNIST_NUMC/modules/MNIST/dataset/t10k-labels.idx1-ubyte"), num_test_labels);
    std::cout<<"\n[+] Test labels fetched!"<<std::endl;
    std::cout<<"-----------Num Labels: "<<num_test_labels<<"------------------------"<<std::endl;


    // displaying random image out of train set.
    random_idx = (rand() % num_test_images); 
    // since it is a 2d array, need to skip 784 (img_size) elements per row.
    showMNIST(test_imgs + random_idx * img_size, 28, 28, std::string(std::string("Ex. test img: ") + std::to_string(test_labels[random_idx])));

    std::cout<<std::endl<<"[.] Starting conversion to float."<<std::endl;

    float *test_imgs_cpu = (float*)malloc(num_test_images * img_size * sizeof(float));
    int* test_labels_cpu = (int*)malloc(num_test_images * sizeof(int));

    for (int i = 0; i < num_test_images; ++i) {

        for (int img_idx = 0; img_idx < img_size; ++img_idx)
            test_imgs_cpu[i * img_size + img_idx] = test_imgs[i * img_size + img_idx];


        test_labels_cpu[i] = test_labels[i];
    }

    free(test_imgs);
    free(test_labels);

    std::cout<<"[+] Done."<<std::endl;

    return {{train_imgs_cpu, val_imgs_cpu, test_imgs_cpu}, {train_labels_cpu, val_labels_cpu, test_labels_cpu}};
}

NeuralNet trainModel(float *x_train, int *y_train, int train_size, float *x_val, int *y_val, int val_size, float *x_test, int *y_test, int test_size, int img_size){
    const int batch_size = 512;
    const int num_epochs = 20;
    NeuralNet best_model;
    NeuralNet model(0, 0.7315);

    int num_iters = ceil(train_size, batch_size);

    std::cout<<std::endl<<"[.] Moving train set to GPU"<<std::endl;
    auto x_train_gpu = np::ArrayGPU<float>(train_size, img_size);
    x_train_gpu.copyFromCPU(x_train);

    auto y_train_gpu = np::ArrayGPU<int>(train_size, 1);
    y_train_gpu.copyFromCPU(y_train);
    std::cout<<"[+] Done."<<std::endl;


    std::cout<<std::endl<<"[.] Converting to batches"<<std::endl;
    auto x_train_batches = np::array_split(x_train_gpu, num_iters, 0);
    auto y_train_batches = np::array_split(y_train_gpu, num_iters, 0);
    std::cout<<"[+] Done."<<std::endl;

    // clearing out of ram
    x_train_gpu = np::zeros<float>(1);
    y_train_gpu = np::zeros<int>(1); 

    std::cout<<std::endl<<"[.] Moving validation set to GPU"<<std::endl;
    auto x_val_gpu = np::ArrayGPU<float>(val_size, img_size);
    x_val_gpu.copyFromCPU(x_val);

    auto y_val_gpu = np::ArrayGPU<int>(val_size, 1);
    y_val_gpu.copyFromCPU(y_val);
    std::cout<<"[+] Done."<<std::endl;

    // for storing the best model
    float best_val_acc= 0, best_train_acc = 0;

    std::cout<<std::endl<<"############### Train Parameters: ###############"<<std::endl;
    std::cout<<"## Network Architecture: ["<<model.affine_layers[0].W.rows<<", "<<model.affine_layers[1].W.cols<<"], ["<<model.affine_layers[1].W.rows<<", "<<model.affine_layers[1].W.cols<<"] ##"<<std::endl;
    std::cout<<"## Initialisation: Xavier Init                 ##"<<std::endl;
    std::cout<<"## Dropout Probablity: "<<model.dropout_layers[0].p_keep<<"                  ##"<<std::endl;
    std::cout<<"## Batch Size: "<<batch_size<<"                             ##"<<std::endl;
    std::cout<<"## Learning Rate: "<<model.adam_configs[0].learning_rate<<"                        ##"<<std::endl;
    std::cout<<"## Adam Params => beta1 = "<<model.adam_configs[0].beta1<<", beta2 = "<<model.adam_configs[0].beta2<<"   ##"<<std::endl;
    std::cout<<"#################################################"<<std::endl;


    std::cout<<std::endl<<"----------------------------------------------------"<<std::endl;
    std::cout<<"------------Beginning Network Training--------------"<<std::endl;
    std::cout<<"----------------------------------------------------"<<std::endl;



    float val_acc = 0, train_acc = 0;
    for(int epoch = 0; epoch< num_epochs; ++epoch){
        for(int iter = 0; iter< num_iters; ++iter){
            // converting to train mode
            model.train();
            
            auto outNloss = model(x_train_batches[iter], y_train_batches[iter]); 

            model.adamStep();

            if( (iter + 1)%100 == 0){
                auto predicted_gpu = outNloss.first.argmax(1);
                auto train_acc = static_cast<float>(((predicted_gpu == y_train_batches[iter]).sum()).at(0))/y_train_batches[iter].rows;

                // evaluating on validation set
                model.eval();
                auto y_pred_gpu = model(x_val_gpu);
                predicted_gpu = y_pred_gpu.argmax(1);

                val_acc = static_cast<float>(((predicted_gpu == y_val_gpu).sum()).at(0))/val_size;

                std::cout<<"Epoch: "<<epoch+1<<" iter: "<<iter+1<<" loss: "<<outNloss.second<<" train_acc: "<<train_acc<<" val_acc: "<<val_acc<<std::endl;

            }
        }
        if( (best_val_acc < val_acc) || (best_val_acc == val_acc && best_train_acc < train_acc) ){
            best_val_acc = val_acc;
            best_train_acc = train_acc;
            best_model = model;
            std::cout<<std::endl<<"##################### NEW BEST FOUND! ###########################"<<std::endl;
            std::cout<<"##################### VAL ACC: "<<std::fixed<<std::setprecision(3)<<best_val_acc<<"                           ##"<<std::endl;
            std::cout<<"##################### TRAIN ACC: "<<std::fixed<<std::setprecision(3)<<train_acc<<"                         ##"<<std::endl;
            std::cout<<"#################################################################"<<std::endl<<std::endl;
        }
    }

    std::cout<<std::endl<<"[+] Model Training Done."<<std::endl;

    std::cout<<std::endl<<"[.] Loading test set on GPU"<<std::endl;
    auto x_test_gpu = np::ArrayGPU<float>(test_size, img_size);
    x_test_gpu.copyFromCPU(x_test);

    auto y_test_gpu = np::ArrayGPU<int>(test_size, 1);
    y_test_gpu.copyFromCPU(y_test);
    std::cout<<"[+] Done."<<std::endl;

    std::cout<<std::endl<<"[.] Performing analysis on test set"<<std::endl;
    best_model.eval();
    auto y_pred_gpu = best_model(x_test_gpu);
    auto predicted_gpu = y_pred_gpu.argmax(1); 
    
    auto test_acc = static_cast<float>(((predicted_gpu == y_test_gpu).sum()).at(0)) / y_pred_gpu.rows;
    std::cout<<"[+] Done."<<std::endl;

    std::cout<<std::endl<<"####### Final model stats: #######"<<std::endl;
    std::cout<<"####### TRAIN ACC: "<<std::fixed<<std::setprecision(3)<<best_train_acc<<"        ##"<<std::endl;
    std::cout<<"####### VAL ACC: "<<std::fixed<<std::setprecision(3)<<best_val_acc<<"          ##"<<std::endl;
    std::cout<<"####### TEST ACC: "<<std::fixed<<std::setprecision(3)<<test_acc<<"         ##"<<std::endl;
    std::cout<<"##################################"<<std::endl;

    return best_model;
}
