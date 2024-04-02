#ifndef ERRCHECKUTILS_H
#define ERRCHECKUTILS_H

#include <curand.h>

#include <iostream>

// cuda error checking macro
#define CUDA_CALL(x)                                                                     \
    do                                                                                   \
    {                                                                                    \
        if ((x) != cudaSuccess)                                                          \
        {                                                                                \
            cudaError_t err = (x);                                                       \
            printf("Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        }                                                                                \
    } while (0)

// curand error checking macro
#define CURAND_CALL(x)                                      \
    do                                                      \
    {                                                       \
        if ((x) != CURAND_STATUS_SUCCESS)                   \
        {                                                   \
            printf("Error at %s:%d\n", __FILE__, __LINE__); \
        }                                                   \
    } while (0)

#endif