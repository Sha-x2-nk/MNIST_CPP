#include <numC/gpuConfig.cuh>

#include <cuda_runtime.h>

#include <cublas_v2.h>

namespace np{
    // getting GPU Config to launch kernels with the most optimal
    int GPU_NUM_CUDA_CORE = 0;
    int GPU_NUM_SM = 0;
    cublasHandle_t cbls_handle;

    int _ConvertSMVer2Cores(int major, int minor){
        // Refer to the CUDA Compute Capability documentation for the number of cores per multiprocessor
        // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
        switch ((major << 4) + minor) {
            case 0x10: return 8;  // Tesla
            case 0x11: return 8;  // Tesla
            case 0x12: return 8;  // Tesla
            case 0x13: return 8;  // Tesla
            case 0x20: return 32; // Fermi
            case 0x21: return 48; // Fermi
            case 0x30: return 192; // Kepler
            case 0x32: return 192; // Kepler
            case 0x35: return 192; // Kepler
            case 0x37: return 192; // Kepler
            case 0x50: return 128; // Maxwell
            case 0x52: return 128; // Maxwell
            case 0x53: return 128; // Maxwell
            case 0x60: return 64;  // Pascal
            case 0x61: return 128; // Pascal
            case 0x62: return 128; // Pascal
            case 0x70: return 64;  // Volta
            case 0x72: return 64;  // Volta
            case 0x75: return 64;  // Turing
            case 0x80: return 64;  // Ampere
            case 0x86: return 128;  // Ampere
            default: return -1;    // Unknown
    }
    }
    void getGPUConfig(int devId){
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, devId);

        GPU_NUM_SM = deviceProp.multiProcessorCount;
        GPU_NUM_CUDA_CORE = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);

        cublasCreate(&cbls_handle);
        
    }
}