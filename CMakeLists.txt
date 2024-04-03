cmake_minimum_required(VERSION 3.22)

project(MNIST_C LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

find_package(OpenCV REQUIRED)

set(SOURCES 
    main.cu
    src/neuralNet.cu
    modules/numC/src/gpuConfig.cu
    modules/layers/src/affineLayer.cu
    modules/layers/src/dropoutLayer.cu
    modules/layers/src/layer.cu
    modules/layers/src/reluLayer.cu
    modules/lossFunctions/src/softmax.cu
    modules/optimisers/src/adam.cu
    modules/MNIST/src/readMNIST.cpp
    modules/MNIST/src/showMNIST.cpp
)

include_directories(include
                    modules/layers/include
                    modules/numC/include                    
                    modules/lossFunctions/include                    
                    modules/optimisers/include                    
                    modules/MNIST/include                    
)

add_executable(main ${SOURCES})

set_target_properties( main PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/../bin
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/../bin
)

target_link_libraries(main cublas ${OpenCV_LIBS})
