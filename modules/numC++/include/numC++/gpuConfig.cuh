#ifndef CUDA_CONFIG_CUH
#define CUDA_CONFIG_CUH

#include <cublas_v2.h>

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