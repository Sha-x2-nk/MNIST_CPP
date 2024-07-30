#ifndef NPFUNCTIONS_CUH
#define NPFUNCTIONS_CUH

#include <numC++/npGPUArray.cuh>
#include <numC++/customKernels.cuh>
#include <numC++/gpuConfig.cuh>
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

	// numbers from [0, range) spaced by 1
	template <typename TP>
	ArrayGPU<TP> arange(const int range);

	// numbers from [start, stop) spaced by step
	template <typename TP>
	ArrayGPU<TP> arange(const float start, const float stop, const float step = 1);

	template <typename TP, Operation OP>
	ArrayGPU<TP> _maxmin(const ArrayGPU<TP> &A, const ArrayGPU<TP> &B);

	template <typename TP, Operation OP>
	ArrayGPU<TP> _maxmin(const ArrayGPU<TP> &A, const TP Scalar);

	// max(a, b). element wise maximum
	template <typename TP>
	ArrayGPU<TP> maximum(const ArrayGPU<TP> &A, const ArrayGPU<TP> &B);

	template <typename TP>
	ArrayGPU<TP> maximum(const ArrayGPU<TP> &A, const TP Scalar);

	template <typename TP>
	ArrayGPU<TP> maximum(const TP Scalar, const ArrayGPU<TP> &A);

	// min(a, b). element wise minimum
	template <typename TP>
	ArrayGPU<TP> minimum(const ArrayGPU<TP> &A, const ArrayGPU<TP> &B);

	template <typename TP>
	ArrayGPU<TP> minimum(const ArrayGPU<TP> &A, const TP Scalar);

	template <typename TP>
	ArrayGPU<TP> minimum(const TP Scalar, const ArrayGPU<TP> &A);

	/*
	functions per element
	F
		#define NP_F_EXP 17
		#define NP_F_LOG 18
		#define NP_F_SQUARE 19
		#define NP_F_SQRT 20
		#define NP_F_POW 21
	*/
	template <typename TP, Operation OP>
	ArrayGPU<TP> _F(const ArrayGPU<TP> &A);

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
	template <typename TP>
	std::vector<np::ArrayGPU<TP>> array_split(const ArrayGPU<TP> &A, const int num_parts, int axis = 0);

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
		dim3 grid(std::min<int>(GPU_NUM_SM * 2, np_ceil(range, block.x)));

		kernelInitMatArange<TP><<<grid, block>>>(ans.mat, range);
		cudaDeviceSynchronize();

		return ans;
	}

	template <typename TP>
	ArrayGPU<TP> arange(const float start, const float stop, const float step)
	{
		const int sz = static_cast<int>((stop - start) / step);
		return start + (arange<TP>(sz) * step);
	}

	/*
	maxmin
	F
		#define NP_OP_MAXIMUM 11
		#define NP_OP_MINIMUM 12
	*/
	template <typename TP, Operation OP>
	ArrayGPU<TP> _maxmin(const ArrayGPU<TP> &A, const ArrayGPU<TP> &B)
	{
		if (A.rows() == 1 && A.cols() == 1)
		{
			// A is scalar
			ArrayGPU<TP> res(B.rows(), B.cols());

			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(std::min<int>(GPU_NUM_SM * 2, np_ceil(res.size(), block.x)));

			kernelMatOpScalar<TP, OP><<<grid, block>>>(B.mat, A.mat, res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}
		else if (B.rows() == 1 && B.cols() == 1)
		{
			// B is scalar
			ArrayGPU<TP> res(A.rows(), A.cols());

			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(std::min<int>(GPU_NUM_SM * 2, np_ceil(res.size(), block.x)));

			kernelMatOpScalar<TP, OP><<<grid, block>>>(A.mat, B.mat, res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}
		// A is vector
		// A vector ki dim, is equal to either col or row of B
		// row vector. will extend along cols if possible. (prioritising in case of square matrix)
		// vice versa for cols

		else if ((A.cols() == 1 && A.rows() == B.rows()) || (A.rows() == 1 && A.cols() == B.rows()))
		{
			// along rows add kr
			ArrayGPU<TP> res(B.rows(), B.cols());
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(std::min<int>(GPU_NUM_SM * 2, np_ceil(res.size(), block.x)));
			kernelMatOpVecAlongCols<TP, OP><<<grid, block>>>(A.mat, B.mat, res.mat, res.size(), B.cols());
			cudaDeviceSynchronize();

			return res;
		}
		else if ((A.cols() == 1 && A.rows() == B.cols()) || (A.rows() == 1 && A.cols() == B.cols()))
		{
			// along cols add kr
			ArrayGPU<TP> res(B.rows(), B.cols());
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(std::min<int>(GPU_NUM_SM * 2, np_ceil(res.size(), block.x)));
			kernelMatOpVecAlongRows<TP, OP><<<grid, block>>>(A.mat, B.mat, res.mat, res.size(), B.cols());
			cudaDeviceSynchronize();

			return res;
		}
		// B is vetor
		// B vector ki dim, is eq to either col or row of B
		// row vector. will extend along cols if possible. (prioritising in case of square matrix)
		else if ((B.cols() == 1 && A.rows() == B.rows()) || (B.rows() == 1 && A.rows() == B.cols()))
		{
			// along rows add kr
			ArrayGPU<TP> res(A.rows(), A.cols());
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(std::min<int>(GPU_NUM_SM * 2, np_ceil(res.size(), block.x)));
			kernelMatOpVecAlongCols<TP, OP><<<grid, block>>>(A.mat, B.mat, res.mat, res.size(), A.cols());
			cudaDeviceSynchronize();

			return res;
		}
		else if ((B.cols() == 1 && A.cols() == B.rows()) || (B.rows() == 1 && A.cols() == B.cols()))
		{
			// along cols add kr
			ArrayGPU<TP> res(A.rows(), A.cols());
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(std::min<int>(GPU_NUM_SM * 2, np_ceil(res.size(), block.x)));
			kernelMatOpVecAlongRows<TP, OP><<<grid, block>>>(A.mat, B.mat, res.mat, res.size(), A.cols());
			cudaDeviceSynchronize();

			return res;
		}
		else if (A.rows() == B.rows() && A.cols() == B.cols())
		{
			// A and B both are matrices of same dimensions
			ArrayGPU<TP> res(A.rows(), A.cols());
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(std::min<int>(GPU_NUM_SM*2, np_ceil(res.size(), block.x)));
			kernelMatOpMat<TP, OP><<<grid, block>>>(A.mat, B.mat, res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}
		else
		{
			std::cerr << "\nError in maximum! Check arguments";
			return np::ArrayGPU<TP>(1, 1, 0);
		}
	}

	template <typename TP, Operation OP>
	ArrayGPU<TP> _maxmin(const ArrayGPU<TP> &A, const TP Scalar)
	{
		ArrayGPU<TP> tmp(1, 1, Scalar);
		return _maxmin<TP, OP>(A, tmp);
	}

	// np.minimum
	template <typename TP>
	ArrayGPU<TP> minimum(const ArrayGPU<TP> &A, const ArrayGPU<TP> &B)
	{
		return _maxmin<TP, NP_OP_MINIMUM>(A, B);
	}

	template <typename TP>
	ArrayGPU<TP> minimum(const ArrayGPU<TP> &A, const TP Scalar)
	{
		return _maxmin<TP, NP_OP_MINIMUM>(A, Scalar);
	}

	template <typename TP>
	ArrayGPU<TP> minimum(const TP Scalar, const ArrayGPU<TP> &A)
	{
		return _maxmin<TP, NP_OP_MINIMUM>(A, Scalar);
	}

	// np.maximum
	template <typename TP>
	ArrayGPU<TP> maximum(const ArrayGPU<TP> &A, const ArrayGPU<TP> &B)
	{
		return _maxmin<TP, NP_OP_MAXIMUM>(A, B);
	}

	template <typename TP>
	ArrayGPU<TP> maximum(const ArrayGPU<TP> &A, const TP Scalar)
	{
		return _maxmin<TP, NP_OP_MAXIMUM>(A, Scalar);
	}

	template <typename TP>
	ArrayGPU<TP> maximum(const TP Scalar, const ArrayGPU<TP> &A)
	{
		return _maxmin<TP, NP_OP_MAXIMUM>(A, Scalar);
	}

	/*
	functions per element
	F
		#define NP_F_EXP 19
		#define NP_F_LOG 20
		#define NP_F_SQUARE 21
		#define NP_F_SQRT 22
	*/
	template <typename TP, Operation OP>
	ArrayGPU<TP> _F(const ArrayGPU<TP> &A)
	{
		ArrayGPU<TP> res(A.rows(), A.cols());

		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(std::min<int>(GPU_NUM_SM * 2, np_ceil(res.size(), block.x)));
		kernelFMat<TP, OP><<<grid, block>>>(A.mat, res.mat, res.size());
		cudaDeviceSynchronize();
		return res;
	}

	// np.exp
	template <typename TP>
	ArrayGPU<TP> exp(const ArrayGPU<TP> &A)
	{
		return _F<TP, NP_F_EXP>(A);
	}

	// np.log
	template <typename TP>
	ArrayGPU<TP> log(const ArrayGPU<TP> &A)
	{
		return _F<TP, NP_F_LOG>(A);
	}

	// np.square
	template <typename TP>
	ArrayGPU<TP> square(const ArrayGPU<TP> &A)
	{
		return _F<TP, NP_F_SQUARE>(A);
	}

	// np.sqrt
	template <typename TP>
	ArrayGPU<TP> sqrt(const ArrayGPU<TP> &A)
	{
		return _F<TP, NP_F_SQRT>(A);
	}

	// np.pow
	template <typename TP>
	ArrayGPU<TP> pow(const ArrayGPU<TP> &A, const float pow)
	{
		ArrayGPU<TP> res(A.rows(), A.cols());

		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(std::min<int>(GPU_NUM_SM * 2, np_ceil(res.size(), block.x)));
		
		kernelPowMat<TP><<<grid, block>>>(A.mat, pow, res.mat, res.size());
		cudaDeviceSynchronize();
		return res;
	}

	// np.shuffle
	template <typename TP>
	void shuffle(ArrayGPU<TP> &A, unsigned long long seed)
	{
		if(A.size() <= 1) return;
		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(np_ceil(A.size(), block.x * 4));

		kernelMatShuffle<TP><<<grid, block>>>(A.mat, A.size(), seed);
		cudaDeviceSynchronize();
	}

	template <typename TP>
	std::vector<np::ArrayGPU<TP>> array_split(const ArrayGPU<TP> &A, const int num_parts, int axis)
	{
		if(num_parts <= 0){
			std::cerr << "INVALID NUM_PARTS ARGUMENT IN array_split!\n";
			return {};
		}
		else if(axis != 0){
			std::cerr << "INVALID AXIS ARGUMENT IN array_split!\n";
			return {};
		}
		else if(num_parts == 1)
			return {A};
		// returns length % n sub-arrays of size length/n + 1 and the rest of size length/n.
		else if (axis == 0)
		{
			std::vector<np::ArrayGPU<TP>> splitted_arrays;

			int tot_size = A.rows();
			int part_size = tot_size / num_parts;
			int remainder = tot_size % num_parts;

			int st_idx = 0;
			for (int i = 0; i < num_parts; ++i)
			{
				int this_part_size = part_size + (i < remainder ? 1 : 0);

				np::ArrayGPU<TP> tmp(A.mat + st_idx, this_part_size, A.cols(), "gpu");

				splitted_arrays.push_back(tmp);

				st_idx += tmp.size();
			}
			return splitted_arrays;
		}
	}

}
#endif