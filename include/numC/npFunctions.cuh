#ifndef NPFUNCTIONS_H
#define NPFUNCTIONS_H

#include <numC/npGPUArray.cuh>
#include <numC/customKernels.cuh>
#include <numC/gpuConfig.cuh>
#include <time.h>

#include <cuda_runtime.h>

#include <iostream>

namespace np
{
	template <typename TP>
	ArrayGPU<TP> ones(const int rows = 1, const int cols = 1);

	template <typename TP>
	ArrayGPU<TP> zeros(const int rows = 1, const int cols = 1);

	template <typename TP>
	ArrayGPU<TP> arange(const int range);

	// max(a, b). element wise maximum
	template <typename TP>
	ArrayGPU<TP> maximum(const ArrayGPU<TP> &A, const ArrayGPU<TP> &B);

	template <typename TP>
	ArrayGPU<TP> maximum(const ArrayGPU<TP> &A, const TP Scalar);

	// np.exp
	template <typename TP>
	ArrayGPU<TP> exp(const ArrayGPU<TP> &A);

	// np.log
	template <typename TP>
	ArrayGPU<TP> log(const ArrayGPU<TP> &A);

	// np.square
	template <typename TP>
	ArrayGPU<TP> square(const ArrayGPU<TP> &A);

	// np.sqrt
	template <typename TP>
	ArrayGPU<TP> sqrt(const ArrayGPU<TP> &A);

	// np.pow
	template <typename TP>
	ArrayGPU<TP> pow(const ArrayGPU<TP> &A, const int pow);

	// np.shuffle
	template <typename TP>
	void shuffle(ArrayGPU<TP> &A, unsigned long long seed = static_cast<unsigned long long>(time(NULL)));

	template <typename TP>
	ArrayGPU<TP> ones(const int rows, const int cols)
	{
		return ArrayGPU<TP>(rows, cols, static_cast<TP>(1));
	}

	template <typename TP>
	ArrayGPU<TP> zeros(const int rows, const int cols)
	{
		return ArrayGPU<TP>(rows, cols, static_cast<TP>(0));
	}

	template <typename TP>
	ArrayGPU<TP> arange(const int range)
	{
		ArrayGPU<TP> ans(range);

		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(ceil(range, block.x));

		kernelInitMatArange<TP><<<grid, block>>>(ans.mat, range);
		cudaDeviceSynchronize();

		return ans;
	}

	template <typename TP>
	// max(a, b). element wise maximum
	ArrayGPU<TP> maximum(const ArrayGPU<TP> &A, const ArrayGPU<TP> &B)
	{
		if (A.rows == 1 || A.cols == 1)
		{
			// A is a scalar
			ArrayGPU<TP> res(B.rows, B.cols);

			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatMaximumScalar<TP><<<grid, block>>>(B.mat, A.at(0), res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}
		else if (B.rows == 1 || B.cols == 1)
		{
			// B is a scalar
			ArrayGPU<TP> res(A.rows, A.cols);

			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatMaximumScalar<TP><<<grid, block>>>(A.mat, B.at(0), res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}
		else if (A.rows == B.rows && A.cols == B.cols)
		{
			// same dimension. element wise comparison

			ArrayGPU<TP> res(A.rows, A.cols);

			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));

			kernelMatMaximumMat<TP><<<grid, block>>>(A.mat, B.mat, res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}
	}

	template <typename TP>
	ArrayGPU<TP> maximum(const ArrayGPU<TP> &A, const TP Scalar)
	{
		ArrayGPU<TP> res(A.rows, A.cols);

		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(ceil(res.size(), block.x));
		kernelMatMaximumScalar<TP><<<grid, block>>>(A.mat, Scalar, res.mat, res.size());
		cudaDeviceSynchronize();
		return res;
	}

	template <typename TP>
	// max(a, b). element wise maximum
	ArrayGPU<TP> minimum(const ArrayGPU<TP> &A, const ArrayGPU<TP> &B)
	{
		if (A.rows == 1 || A.cols == 1)
		{
			// A is a scalar
			ArrayGPU<TP> res(B.rows, B.cols);

			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatMinimumScalar<TP><<<grid, block>>>(B.mat, A.at(0), res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}
		else if (B.rows == 1 || B.cols == 1)
		{
			// B is a scalar
			ArrayGPU<TP> res(A.rows, A.cols);

			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatMinimumScalar<TP><<<grid, block>>>(A.mat, B.at(0), res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}
		else if (A.rows == B.rows && A.cols == B.cols)
		{
			// same dimension. element wise comparison

			ArrayGPU<TP> res(A.rows, A.cols);

			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));

			kernelMatMinimumMat<TP><<<grid, block>>>(A.mat, B.mat, res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}
	}

	template <typename TP>
	ArrayGPU<TP> minimum(const ArrayGPU<TP> &A, const TP Scalar)
	{
		ArrayGPU<TP> res(A.rows, A.cols);

		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(ceil(res.size(), block.x));
		kernelMatMinimumScalar<TP><<<grid, block>>>(A.mat, Scalar, res.mat, res.size());
		cudaDeviceSynchronize();
		return res;
	}

	// np.exp
	template <typename TP>
	ArrayGPU<TP> exp(const ArrayGPU<TP> &A)
	{
		ArrayGPU<TP> res(A.rows, A.cols);

		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(ceil(res.size(), block.x));
		kernelExpMat<TP><<<grid, block>>>(A.mat, res.mat, res.size());
		cudaDeviceSynchronize();
		return res;
	}

	// np.log
	template <typename TP>
	ArrayGPU<TP> log(const ArrayGPU<TP> &A)
	{
		ArrayGPU<TP> res(A.rows, A.cols);

		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(ceil(res.size(), block.x));
		kernelLogMat<TP><<<grid, block>>>(A.mat, res.mat, res.size());
		cudaDeviceSynchronize();
		return res;
	}

	// np.square
	template <typename TP>
	ArrayGPU<TP> square(const ArrayGPU<TP> &A)
	{
		ArrayGPU<TP> res(A.rows, A.cols);

		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(ceil(res.size(), block.x));
		kernelSquareMat<TP><<<grid, block>>>(A.mat, res.mat, res.size());
		cudaDeviceSynchronize();
		
		return res;
	}

	// np.sqrt
	template <typename TP>
	ArrayGPU<TP> sqrt(const ArrayGPU<TP> &A)
	{
		ArrayGPU<TP> res(A.rows, A.cols);

		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(ceil(res.size(), block.x));
		kernelSqrtMat<TP><<<grid, block>>>(A.mat, res.mat, res.size());
		cudaDeviceSynchronize();
		return res;
	}

	// np.pow
	template <typename TP>
	ArrayGPU<TP> pow(const ArrayGPU<TP> &A, const int pow)
	{
		ArrayGPU<TP> res(A.rows, A.cols);

		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(ceil(res.size(), block.x));
		kernelPowMat<TP><<<grid, block>>>(A.mat, pow, res.mat, res.size());
		cudaDeviceSynchronize();
		return res;
	}

	// np.shuffle 
	template <typename TP>
	void shuffle(ArrayGPU<TP> &A, unsigned long long seed){
		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(ceil(A.size(), block.x));

		kernelMatShuffle<TP><<<grid, block>>>(A.mat, A.size(), seed); 
		cudaDeviceSynchronize();
	}
}
#endif