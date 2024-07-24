#ifndef ERRCHECKUTILS_CUH
#define ERRCHECKUTILS_CUH

// cuda includes
#include <cuda_runtime.h>
#include <curand.h>

// std includes
#include <stdio.h>

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

namespace np{
    #define np_ceil(x, y) ((x + y - 1) / y)
	enum Operation {
		NP_OP_ADD,
		NP_OP_SUB,
		NP_OP_MUL,
		NP_OP_DIV,
		NP_OP_LESS_THAN,
		NP_OP_LESS_THAN_EQ,
		NP_OP_GREATER_THAN,
		NP_OP_GREATER_THAN_EQ,
		NP_OP_EQ_EQ,
		NP_OP_NOT_EQ,
		NP_OP_MINIMUM,
		NP_OP_MAXIMUM,
		NP_OP_EQ,

		NP_REDUCE_SUM,
		NP_REDUCE_MIN,
		NP_REDUCE_MAX,
		NP_REDUCE_ARGMIN,
		NP_REDUCE_ARGMAX,

		NP_F_EXP,
		NP_F_LOG,
		NP_F_SQUARE,
		NP_F_SQRT,
		NP_F_POW
	};
}

#endif