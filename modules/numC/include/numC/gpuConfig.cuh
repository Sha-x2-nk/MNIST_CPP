#ifndef CUDA_CONFIG_H
#define CUDA_CONFIG_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

namespace np
{
    // getting GPU Config to launch kernels with the most optimal
    extern int GPU_NUM_CUDA_CORE;
    extern int GPU_NUM_SM;
    extern cublasHandle_t cbls_handle;

    int _ConvertSMVer2Cores(int major, int minor);
    void getGPUConfig(int devId = 0);
}
#endif