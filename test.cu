#include <cuda_runtime.h>
#include <iostream>

__global__ void kernelAdd1(const float *A, const float *B, float *C, const int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < size){
        C[idx] = A[idx] + B[idx];
    }
}
int main(){
    return 0;
}