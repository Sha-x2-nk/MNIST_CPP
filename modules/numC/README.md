# numC: Accelerated GPU Library for C++, with NumPy syntax
numC is a C++ library inspired by numpy, designed for accelerated numerical computations on GPUs. It provides a familiar syntax similar to numpy, making it easy for users familiar with numpy to transition to GPU-accelerated computing seamlessly while working on C++.

Currently, numC only supports 2D arrays.

## Installation
To use numC, follow these steps:

1. Clone the repository:

    `git clone https://github.com/Sha-x2-nk/numC.git`
2. Include whatever numC headers / functions you require:
    ```cpp 
    #include "numC/npGPUArray.cuh"
    ```

3. compile you numC including program using nvcc, and also compile and link gpuConfig.cu file.

## Features
### ArrayGPU Class
The ArrayGPU class provides functionalities for creating, manipulating, and performing operations on GPU-accelerated arrays. Here are some key features:

* Creation and initialization of arrays

```cpp
#include "numC/npGPUArray.cuh"
#include "numC/gpuConfig.cuh"

int main(){
    np::getGPUConfig(0); 
    /* mandatory call, once in main function. this function gets gpu config, i.e. number of cuda cores and number of SMs to efficiently schedule algorithms on GPU. */

    auto A = np::ArrayGPU<float>(10, 2); // creates a 10x2 array filled with 0s
    A = np::ArrayGPU<float>(10, 2, 5); // 10x2 array with all elements as 5.

    int r = A.rows; // gets number of rows
    int c = A.cols; // gets number of cols
    int sz = A.size(); // gets size, rows * cols
    float *m = A.mat; // pointer pointing to array in global memory of GPU
}
```
* Reshaping arrays
```cpp
    A.reshape(5, 4); // reshapes it to 5x4 array. reshape fails when new size does not match old size
```
* Getter, setter
```cpp
auto A = np::ArrayGPU<float>(10, 5);
float v = A.at(0); // assumes array as linear and gives number at position. 
v = A.at(5, 4); // gives number at 5, 4 position.
// above functions return the value as typename(float in this case), and the variable is transferred to CPU RAM. Hence printing by direct for loop will take time - each variable will be moved to CPU RAM from GPU RAM, numC has print and cout for that..

// numC also has support for indexing multiple elements at once, like numpy. This will become better when we introduce np::arange function

auto C = A.at(np::ArrayGPU<int> idxs); // will return a 1D ArrayGPU with all the elements from idxs given as parameter. order maintained.
// Ci = Ai where i is from idxs.

auto C = A.at(np::ArrayGPU<int> row, np::ArrayGPU<int> col); // Ci = A(rowi, coli).

// set function also has similar APIs
C.set(0, 5); // sets 0th index element as 5
C.set(np::ArrayGPU<int> idxs, np::ArrayGPU<float> val); 
C.set(np::ArrayGPU<int> rows, np::ArrayGPU<int> cols, np::ArrayGPU<float> val);
```
* print function.
```cpp
auto A = np::ArrayGPU<float>(1024, 1024);
A.print(); // prints whole array
std::cout<<A; // numC has overloaded << operator with cout, so cout also prints the full array.
```
* Element-wise operations (addition, subtraction, multiplication, division). Returns a new array. (Supports broadcasting)
```cpp
    auto A = np::ArrayGPU<float>(10, 2); 
    auto B = np::ArrayGPU<float>(10, 2);
    // currently only same type arrays are supported for operators.
    auto C = A + B;

    // broadcasting
    C = A + 5;
    C = A + np::ArrayGPU<float>(1, 2); 
    C = A + np::ArrayGPU<float>(10, 1);
    C = A + np::ArrayGPU<float>(1, 1, 0);

    // shown for +, but also works with -, *, / operators
```
* Comparison operations (>, <, >=, <=, ==, !=). Returns array of 0s and 1s, depending on condition. (Supports broadcasting)
```cpp
    auto A = np::ArrayGPU<float>(10, 2); 
    auto B = np::ArrayGPU<float>(10, 2);
    // currently only same type arrays are supported for operators.
    auto C = A < B;

    // broadcasting
    C = A < 5;
    C = A < np::ArrayGPU<float>(1, 2); 
    C = A < np::ArrayGPU<float>(10, 1);
    C = A < np::ArrayGPU<float>(1, 1, 0);

    // shown for <, but also works with <=, >, >=, ==, != operators
```
* Transpose. returns a new transposed array. Transposition kernel is optimised and delivers ~97% of copy speeds during transposition.
```cpp
    auto A = np::ArrayGPU<float>(10, 2); 
    auto AT = A.T();
```
* dot product
```cpp
auto A = np::ArrayGPU<float>(128, 1024);
auto B = np::ArrayGPU<float>(1024, 128);
auto C = A.dot(B);

// other dot functions
B = np::ArrayGPU<float>(128, 1024);
C = A.Tdot(B); // A transpose dot B

C = A.dotT(B); // A dot B transpose

```
* Statistical functions (sum, max, min, argmax, argmin). Returns a new array. additional argument - axis
```cpp
auto A = np::ArrayGPU<float>(128, 1024);

// default axis = -1. i.e. total sum
auto C = A.sum(); // returns 1x1 array
C = A.sum(0); // column wise sum. returns array of dimension 1x1024
C = A.sum(1); // row wise sum. returns array of dimension 128x1

// works similarly with sum, max, min, argmax, argmin.
// argmax, argmin return indexes of element instead of elements. return type is mandatorily np::ArrayGPU<int> for these functions.
```

### npFunctions header
* ones, zeros and arange
```cpp
#include "numC/npFunctions.cuh"

// call gpuConfig

auto C = np::ones<float>(10); // 1d array of 1s
C = np::ones<float>(10, 10); // 2d array of 1s

C = np::zeros<float>(10, 10); // 1d array of 0s 
C = np::zeros<float>(10, 10); // 2d array of 0s

C = np::arange<float>(10); // 1d array with numbers from 0 to 9, all at their respective indexes. Immensely powerful for collective indexing, as shown earlier
```

* maximum, minimum
```cpp
#include "numC/npFunctions.cuh"

// call gpuConfig
auto A = np::ArrayGPU<float>(10, 5, 7); // 10x5 array, fill with 7
auto B = np::ArrayGPU<float>(10, 5, 6);

auto C = np::maximum(A, B);

// broadcasting
C = np::maximum(A, 0);
C = np::maximum(A, np::ArrayGPU<float>(10, 1));  
C = np::maximum(A, np::ArrayGPU<float>(1, 5));
```
* exp, log, square, sqrt
```cpp


Random array generation (uniform and normal distributions)
