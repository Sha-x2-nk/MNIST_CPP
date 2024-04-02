cmake_minimum_required(VERSION 3.22)

project(MNIST_C LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

find_package(OpenCV REQUIRED)
# set(WITH_CUDA OFF)

set(SOURCES 
    main.cu
    src/numC/gpuConfig.cu
    src/layers/affineLayer.cu
    src/layers/dropoutLayer.cu
    src/layers/layer.cu
    src/layers/reluLayer.cu
    src/lossFunctions/softmax.cu
    src/optimisers/adam.cu
    src/NeuralNet.cu
    src/MNIST/mnist_read.cpp
    src/visualisations/showImg.cpp
)

include_directories(include)

add_executable(main ${SOURCES})

set_target_properties( main PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/../bin
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/../bin
)

target_link_libraries(main cublas)
target_link_libraries(main cublas ${OpenCV_LIBS})
