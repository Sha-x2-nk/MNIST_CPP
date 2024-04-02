#ifndef NPRANDOM_H
#define NPRANDOM_H

#include <numC/npRandom.cuh>
#include <numC/customKernels.cuh>
#include <numC/npGPUArray.cuh>
#include <numC/errorCheckutils.cuh>
#include <numC/gpuConfig.cuh>

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include <iostream>
#include <time.h>

namespace np
{
    class Random
    {
    public:
        // from uniform distribution
        template <typename TP>
        static ArrayGPU<TP> rand(int rows = 1, int cols = 1, int lo = 0, int hi = 1, unsigned long long seed = static_cast<unsigned long long>(time(NULL)));

        template <typename TP>
        static ArrayGPU<TP> rand(int rows, int cols, unsigned long long seed);

        // from normal distribution
        template <typename TP>
        static ArrayGPU<TP> randn(int rows = 1, int cols = 1, unsigned long long seed = static_cast<unsigned long long>(time(NULL)));
    };

    template <typename TP>
    ArrayGPU<TP> Random::rand(int rows, int cols, unsigned long long seed)
    {
        ArrayGPU<TP> ar(rows, cols);

        const int BLOCK_SIZE = 16;
        dim3 block(BLOCK_SIZE * BLOCK_SIZE);
        dim3 grid(ceil(rows * cols, block.x));
        kernelInitializeRandomUnif<TP><<<grid, block>>>(ar.mat, rows * cols, seed);
        cudaDeviceSynchronize();

        return ar;
    }

    template <typename TP>
    ArrayGPU<TP> Random::rand(int rows, int cols, int lo, int hi, unsigned long long seed)
    {
        ArrayGPU<TP> ar(rows, cols);

        const int BLOCK_SIZE = (GPU_NUM_CUDA_CORE == 64) ? 8 : 16;
        dim3 block(BLOCK_SIZE * BLOCK_SIZE);
        dim3 grid(ceil(rows * cols, block.x));
        kernelInitializeRandomUnif<TP><<<grid, block>>>(ar.mat, rows * cols, lo, hi, seed);
        cudaDeviceSynchronize();

        return ar;
    }

    // from normal distribution
    template <typename TP>
    ArrayGPU<TP> Random::randn(int rows, int cols, unsigned long long seed)
    {
        ArrayGPU<TP> ar(rows, cols);

        const int BLOCK_SIZE = 16;
        dim3 block(BLOCK_SIZE * BLOCK_SIZE);
        dim3 grid(ceil(rows * cols, block.x));
        kernelInitializeRandomNorm<TP><<<grid, block>>>(ar.mat, rows * cols, seed);
        cudaDeviceSynchronize();

        return ar;
    }
}

#endif