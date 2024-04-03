#ifndef NPFUNCTIONS_H
#define NPFUNCTIONS_H

#include <numC/npGPUArray.cuh>
#include <numC/customKernels.cuh>
#include <numC/gpuConfig.cuh>
#include <time.h>

#include <cuda_runtime.h>

#include <iostream>
#include <vector>
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

	// min(a, b). element wise minimum
	template <typename TP>
	ArrayGPU<TP> minimum(const ArrayGPU<TP> &A, const ArrayGPU<TP> &B);

	template <typename TP>
	ArrayGPU<TP> minimum(const ArrayGPU<TP> &A, const TP Scalar);


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

	// np.array_split
	template<typename TP>
	std::vector<np::ArrayGPU<TP>> array_split(const np::ArrayGPU<TP> &A, const int num_parts, int axis = 0);

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

	// np.maximum
	template <typename TP>
	ArrayGPU<TP> maximum(const ArrayGPU<TP> &A, const ArrayGPU<TP> &B)
	{
		if (A.rows == 1 && A.cols == 1)
		{
			// A is scalar
			ArrayGPU<TP> res(B.rows, B.cols);

			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));

			kernelMatMaximumScalar<TP><<<grid, block>>>(B.mat, A.at(0), res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}
		else if (B.rows == 1 && B.cols == 1)
		{
			// B is scalar
			ArrayGPU<TP> res(A.rows, A.cols);

			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));

			kernelMatMaximumScalar<TP><<<grid, block>>>(A.mat, B.at(0), res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}
		// A is vector
		// A vector ki dim, is equal to either col or row of B
		// row vector. will extend along cols if possible. (prioritising in case of square matrix)
		// vice versa for cols

		else if ((A.cols == 1 && A.rows == B.rows) || (A.rows == 1 && A.cols == B.rows))
		{
			// along rows add kr
			ArrayGPU<TP> res(B.rows, B.cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatMaximumVecAlongCols<TP><<<grid, block>>>(A.mat, B.mat, res.mat, res.size(), B.cols);
			cudaDeviceSynchronize();

			return res;
		}
		else if ((A.cols == 1 && A.rows == B.cols) || (A.rows == 1 && A.cols == B.cols))
		{
			// along cols add kr
			ArrayGPU<TP> res(B.rows, B.cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatMaximumVecAlongRows<TP><<<grid, block>>>(A.mat, B.mat, res.mat, res.size(), B.cols);
			cudaDeviceSynchronize();

			return res;
		}
		// B is vetor
		// B vector ki dim, is eq to either col or row of B
		// row vector. will extend along cols if possible. (prioritising in case of square matrix)
		else if ((B.cols == 1 && A.rows == B.rows) || (B.rows == 1 && A.rows == B.cols))
		{
			// along rows add kr
			ArrayGPU<TP> res(A.rows, A.cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatMaximumVecAlongCols<TP><<<grid, block>>>(A.mat, B.mat, res.mat, res.size(), A.cols);
			cudaDeviceSynchronize();

			return res;
		}
		else if ((B.cols == 1 && A.cols == B.rows) || (B.rows == 1 && A.cols == B.cols))
		{
			// along cols add kr
			ArrayGPU<TP> res(A.rows, A.cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatMaximumVecAlongRows<TP><<<grid, block>>>(A.mat, B.mat, res.mat, res.size(), A.cols);
			cudaDeviceSynchronize();

			return res;
		}
		else if (A.rows == B.rows && A.cols == B.cols)
		{
			// A and B both are matrices of same dimensions
			ArrayGPU<TP> res(A.rows, A.cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatMaximumMat<TP><<<grid, block>>>(A.mat, B.mat, res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}
		else
		{
			std::cerr << "\nError in maximum! Check arguments";
			return np::ArrayGPU<TP>(1, 1, 0);
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

	
	// np.minimum
	template <typename TP>
	ArrayGPU<TP> minimum(const ArrayGPU<TP> &A, const ArrayGPU<TP> &B)
	{
		if (A.rows == 1 && A.cols == 1)
		{
			// A is scalar
			ArrayGPU<TP> res(B.rows, B.cols);

			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));

			kernelMatMinimumScalar<TP><<<grid, block>>>(B.mat, A.at(0), res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}
		else if (B.rows == 1 && B.cols == 1)
		{
			// B is scalar
			ArrayGPU<TP> res(A.rows, A.cols);

			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));

			kernelMatMinimumScalar<TP><<<grid, block>>>(A.mat, B.at(0), res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}
		// A is vector
		// A vector ki dim, is equal to either col or row of B
		// row vector. will extend along cols if possible. (prioritising in case of square matrix)
		// vice versa for cols

		else if ((A.cols == 1 && A.rows == B.rows) || (A.rows == 1 && A.cols == B.rows))
		{
			// along rows add kr
			ArrayGPU<TP> res(B.rows, B.cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatMinimumVecAlongCols<TP><<<grid, block>>>(A.mat, B.mat, res.mat, res.size(), B.cols);
			cudaDeviceSynchronize();

			return res;
		}
		else if ((A.cols == 1 && A.rows == B.cols) || (A.rows == 1 && A.cols == B.cols))
		{
			// along cols add kr
			ArrayGPU<TP> res(B.rows, B.cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatMinimumVecAlongRows<TP><<<grid, block>>>(A.mat, B.mat, res.mat, res.size(), B.cols);
			cudaDeviceSynchronize();

			return res;
		}
		// B is vetor
		// B vector ki dim, is eq to either col or row of B
		// row vector. will extend along cols if possible. (prioritising in case of square matrix)
		else if ((B.cols == 1 && A.rows == B.rows) || (B.rows == 1 && A.rows == B.cols))
		{
			// along rows add kr
			ArrayGPU<TP> res(A.rows, A.cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatMinimumVecAlongCols<TP><<<grid, block>>>(A.mat, B.mat, res.mat, res.size(), A.cols);
			cudaDeviceSynchronize();

			return res;
		}
		else if ((B.cols == 1 && A.cols == B.rows) || (B.rows == 1 && A.cols == B.cols))
		{
			// along cols add kr
			ArrayGPU<TP> res(A.rows, A.cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatMinimumVecAlongRows<TP><<<grid, block>>>(A.mat, B.mat, res.mat, res.size(), A.cols);
			cudaDeviceSynchronize();

			return res;
		}
		else if (A.rows == B.rows && A.cols == B.cols)
		{
			// A and B both are matrices of same dimensions
			ArrayGPU<TP> res(A.rows, A.cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatMinimumMat<TP><<<grid, block>>>(A.mat, B.mat, res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}
		else
		{
			std::cerr << "\nError in maximum! Check arguments";
			return np::ArrayGPU<TP>(1, 1, 0);
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
	ArrayGPU<TP> pow(const ArrayGPU<TP> &A, const float pow)
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

	template<typename TP>
	std::vector<np::ArrayGPU<TP>> array_split(const np::ArrayGPU<TP> &A, const int num_parts, int axis){
		// returns length % n sub-arrays of size length/n + 1 and the rest of size length/n.
		if(axis == 0){
			std::vector<np::ArrayGPU<TP>> splitted_arrays;

			int tot_size = A.rows;
			int part_size = tot_size / num_parts;
			int remainder = tot_size % num_parts;

			int st_idx = 0;
			for(int i= 0; i< num_parts; ++i){
				int this_part_size = part_size + (i< remainder ? 1: 0);

				np::ArrayGPU<TP> tmp(this_part_size, A.cols);
				tmp.copyFromGPU(A.mat + st_idx);

				splitted_arrays.push_back(tmp);

				st_idx += tmp.size();
			}
			return splitted_arrays;
		}
		else{
			std::cerr<<"INVALID AXIS ARGUMENT!\n";
			return {};
		}
	}
}
#endif