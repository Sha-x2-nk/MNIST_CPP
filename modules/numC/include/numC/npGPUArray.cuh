#ifndef NPGPUARRAY_H
#define NPGPUARRAY_H

#include <numC/npGPUArray.cuh>
#include <numC/customKernels.cuh>
#include <numC/errorCheckUtils.cuh>
#include <numC/gpuConfig.cuh>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>

#define ceil(x, y) (x + y - 1) / y

namespace np
{
	template <typename TP>
	class ArrayGPU
	{
	private:
	public:
		TP *mat;
		int rows, cols;

		ArrayGPU(const int rows = 1, const int cols = 1);

		// initialise array with all values set to Val
		ArrayGPU(const int rows, const int cols, const TP Val);

		ArrayGPU(const ArrayGPU<TP> &A);

		void reshape(const int newRows, const int newCols);

		unsigned int size() const;

		// pointer to host memory.
		void copyFromCPU(TP *h_array);

		// pointer to device memory.
		void copyFromGPU(TP *d_array);

		void print() const;

		// transpose
		ArrayGPU<TP> T() const;

		// get value at an index
		TP at(const int idx) const;

		// get value at r, c
		TP at(const int r, const int c) const;

		// get values from multiple indexes
		ArrayGPU<TP> at(const ArrayGPU<int> &idxs) const;

		// get values from multiple indexes
		ArrayGPU<TP> at(const ArrayGPU<int> &r, const ArrayGPU<int> &c) const;

		// set value at idx
		void set(const int idx, const TP val);

		// set value at r, c
		void set(const int r, const int c, const TP val);

		// set values from multiple indexes
		void set(const ArrayGPU<int> &idxs, const ArrayGPU<TP> &val);

		// set values from multiple indexes
		void set(const ArrayGPU<int> &r, const ArrayGPU<int> &c, const ArrayGPU<TP> &val);

		// defining dot product
		ArrayGPU<TP> dot(const ArrayGPU<TP> &B) const;

		ArrayGPU<TP> Tdot(const ArrayGPU<TP> &B) const;

		ArrayGPU<TP> dotT(const ArrayGPU<TP> &B) const;

		// assignment operator overload
		void operator=(const ArrayGPU<TP> &A);

		// add functions
		ArrayGPU<TP> operator+(const ArrayGPU<TP> &B) const;

		ArrayGPU<TP> operator+(const TP Scalar) const;

		// minus
		ArrayGPU<TP> operator-(const ArrayGPU<TP> &B) const;

		ArrayGPU<TP> operator-(const TP Scalar) const;

		// unary negation operator
		ArrayGPU<TP> operator-() const;

		// multiply
		ArrayGPU<TP> operator*(const ArrayGPU<TP> &B) const;

		ArrayGPU<TP> operator*(const TP Scalar) const;

		// divide
		ArrayGPU<TP> operator/(const ArrayGPU<TP> &B) const;

		ArrayGPU<TP> operator/(const TP Scalar) const;

		// returns an array of 0s and 1s depending on true or false of the conditions.
		//  element wise comparison

		// >
		ArrayGPU<TP> operator>(const ArrayGPU<TP> &B) const;

		ArrayGPU<TP> operator>(const TP Scalar) const;

		// <
		ArrayGPU<TP> operator<(const ArrayGPU<TP> &B) const;

		ArrayGPU<TP> operator<(const TP Scalar) const;

		// >=
		ArrayGPU<TP> operator>=(const ArrayGPU<TP> &B) const;

		ArrayGPU<TP> operator>=(const TP Scalar) const;

		// <=
		ArrayGPU<TP> operator<=(const ArrayGPU<TP> &B) const;

		ArrayGPU<TP> operator<=(const TP Scalar) const;

		// ==
		ArrayGPU<TP> operator==(const ArrayGPU<TP> &B) const;

		ArrayGPU<TP> operator==(const TP Scalar) const;

		// !=
		ArrayGPU<TP> operator!=(const ArrayGPU<TP> &B) const;

		ArrayGPU<TP> operator!=(const TP Scalar) const;

		// sum. along axis or total
		ArrayGPU<TP> sum(const int axis = -1) const;

		// max. along axis or total
		ArrayGPU<TP> max(const int axis = -1) const;

		// min. along axis or total
		ArrayGPU<TP> min(const int axis = -1) const;

		// argmax
		ArrayGPU<int> argmax(const int axis = -1) const;

		// argmin
		ArrayGPU<int> argmin(const int axis = -1) const;

		// sort
		// argsort

		~ArrayGPU();
	};

	template <typename TP>
	ArrayGPU<TP>::ArrayGPU(const int rows, const int cols)
	{
		this->rows = rows;
		this->cols = cols;

		CUDA_CALL(cudaMalloc((void **)&this->mat, this->rows * this->cols * sizeof(TP)));
		CUDA_CALL(cudaMemset(this->mat, 0, this->rows * this->cols * sizeof(TP)));
	}

	// initialise all values with same value (broadcast)
	template <typename TP>
	ArrayGPU<TP>::ArrayGPU(const int rows, const int cols, const TP Val)
	{
		this->rows = rows;
		this->cols = cols;

		CUDA_CALL(cudaMalloc((void **)&this->mat, this->rows * this->cols * sizeof(TP)));

		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(ceil(this->rows * this->cols, block.x));
		kernelInitMatBroadcast<TP><<<grid, block>>>(mat, Val, this->rows * this->cols);
		cudaDeviceSynchronize();
	}

	// copy constructor
	template <typename TP>
	ArrayGPU<TP>::ArrayGPU(const ArrayGPU<TP> &A)
	{
		this->rows = A.rows;
		this->cols = A.cols;
		CUDA_CALL(cudaMalloc((void **)&this->mat, this->rows * this->cols * sizeof(TP)));

		this->copyFromGPU(A.mat);
	}

	template <typename TP>
	void ArrayGPU<TP>::reshape(const int newRows, const int newCols)
	{
		if (newRows * newCols == this->rows * this->cols)
		{
			this->rows = newRows;
			this->cols = newCols;
		}
		else
		{
			std::cerr << "\nError! New size and old size are not equal.";
		}
	}

	template <typename TP>
	unsigned int ArrayGPU<TP>::size() const
	{
		return this->rows * this->cols;
	}

	// pointer to host memory.
	template <typename TP>
	void ArrayGPU<TP>::copyFromCPU(TP *h_array)
	{
		CUDA_CALL(cudaMemcpy(mat, h_array, this->rows * this->cols * sizeof(TP), cudaMemcpyHostToDevice));
	}

	// pointer to device memory.
	template <typename TP>
	void ArrayGPU<TP>::copyFromGPU(TP *d_array)
	{
		CUDA_CALL(cudaMemcpy(this->mat, d_array, this->rows * this->cols * sizeof(TP), cudaMemcpyDeviceToDevice));
	}

	template <typename TP>
	void ArrayGPU<TP>::print() const
	{
		kernelPrintMat<TP><<<1, 1>>>(mat, this->rows, this->cols);
		cudaDeviceSynchronize();
	}

	// overloading cout
	template <typename TP>
	std::ostream &operator<<(std::ostream &out, ArrayGPU<TP> &A)
	{
		A.print();
		return out;
	}

	// transpose
	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::T() const
	{
		ArrayGPU<TP> out(this->cols, this->rows);

		const int TILE_WIDTH = (GPU_NUM_CUDA_CORE == 64) ? 8 : 16;
		const int ROW_BLOCK = (GPU_NUM_CUDA_CORE == 64) ? 4 : 8;
		dim3 block(TILE_WIDTH, ROW_BLOCK);
		dim3 grid(ceil(this->cols, TILE_WIDTH), ceil(this->rows, TILE_WIDTH));

		switch (GPU_NUM_CUDA_CORE)
		{
		case 64:
			kernelTransposeInMem<TP, 8, 4><<<grid, block>>>(this->mat, out.mat, this->rows, this->cols);
			break;

		default:
			kernelTransposeInMem<TP, 16, 8><<<grid, block>>>(this->mat, out.mat, this->rows, this->cols);
			break;
		}
		cudaDeviceSynchronize();

		return out;
	}

	// get value at idx
	template <typename TP>
	TP ArrayGPU<TP>::at(const int idx) const
	{
		TP val;
		CUDA_CALL(cudaMemcpy(&val, mat + idx, sizeof(TP), cudaMemcpyDeviceToHost));
		return val;
	}

	// get value at r, c
	template <typename TP>
	TP ArrayGPU<TP>::at(const int r, const int c) const
	{
		return at(r * this->cols + c);
	}

	// get values from multiple indexes
	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::at(const ArrayGPU<int> &idxs) const
	{
		int size = std::max<int>(idxs.rows, idxs.cols);
		ArrayGPU<TP> ans(size);

		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(ceil(size, block.x));
		kernelGetMatValues<TP><<<grid, block>>>(mat, ans.mat, idxs.mat, size);
		cudaDeviceSynchronize();

		return ans;
	}

	// get values from multiple indexes
	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::at(const ArrayGPU<int> &r, const ArrayGPU<int> &c) const
	{
		/*
			r = (0, 1, 2, 3, 4, 5, 6)
			c = (7, 6, 4, 2, 1, 8, 9)
			fetch all (ri , ci) elements
		*/
		int size = std::max<int>(r.rows, r.cols);
		ArrayGPU<TP> ans(size);

		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(ceil(size, block.x));
		kernelGetMatValues<TP><<<grid, block>>>(mat, this->cols, ans.mat, r.mat, c.mat, size);
		cudaDeviceSynchronize();

		return ans;
	}

	// set value at idx
	template <typename TP>
	void ArrayGPU<TP>::set(const int idx, const TP val)
	{
		CUDA_CALL(cudaMemcpy(mat + idx, &val, sizeof(TP), cudaMemcpyHostToDevice));
	}

	// set value at r, c
	template <typename TP>
	void ArrayGPU<TP>::set(const int r, const int c, const TP val)
	{
		int idx = r * this->cols + c;
		set(idx, val);
	}

	// set values from multiple indexes
	template <typename TP>
	void ArrayGPU<TP>::set(const ArrayGPU<int> &idxs, const ArrayGPU<TP> &val)
	{
		/*
			r = (0, 1, 2, 3, 4, 5, 6)
			c = (7, 6, 4, 2, 1, 8, 9)
			val = (1, 2, 3, 4, 5, 6, 7)
		set all (ri , ci) elements to vali
		*/
		int size = std::max<int>(idxs.rows, idxs.cols); // one dimension will always be 1.

		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(ceil(size, block.x));
		kernelSetMatValues<TP><<<grid, block>>>(mat, val.mat, idxs.mat, size);
		cudaDeviceSynchronize();
	}

	// set values from multiple indexes
	template <typename TP>
	void ArrayGPU<TP>::set(const ArrayGPU<int> &r, const ArrayGPU<int> &c, const ArrayGPU<TP> &val)
	{
		/*
			r = (0, 1, 2, 3, 4, 5, 6)
			c = (7, 6, 4, 2, 1, 8, 9)
			val = (1, 2, 3, 4, 5, 6, 7)
		set all (ri , ci) elements to vali
		*/
		int size = std::max<int>(r.rows, r.cols); // one dimension will always be 1.

		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(ceil(size, block.x));
		kernelSetMatValues<TP><<<grid, block>>>(mat, this->cols, val.mat, r.mat, c.mat, size);
		cudaDeviceSynchronize();
	}

	// defining dot product
	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::dot(const ArrayGPU<TP> &B) const
	{
		/*
			C = A @ B
			input:
				A: shape MxK
				B: shape KxN
			output:
				C: shape MxN

		*/
		// condition for dot product
		if (this->cols == B.rows)
		{
			ArrayGPU<TP> res(this->rows, B.cols);

			const float alpha = 1.0f;
			const float beta = 0.0f;

			// C = A . B k lie.
			cublasSgemm(cbls_handle, //
						CUBLAS_OP_N, CUBLAS_OP_N,
						B.cols, this->rows, this->cols, // B cols, A rows, A cols
						&alpha,
						B.mat, B.cols,		   // B, B cols
						this->mat, this->cols, // A, A cols
						&beta,
						res.mat, B.cols); // C, B cols

			return res;
		}
		else
		{
			std::cerr << "\nError in dot! Check arguments";
			return np::ArrayGPU<TP>(1, 1, 0);
		}
	}

	// dot with first matrix transposed
	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::Tdot(const ArrayGPU<TP> &B) const
	{
		/*
			C = A.T @ B
			input:
				A: shape KxM
				B: shape KxN
			output:
				C: shape MxN

		*/
		// condition for dot product
		if (this->rows == B.rows)
		{
			ArrayGPU<TP> res(this->cols, B.cols);

			const float alpha = 1.0f;
			const float beta = 0.0f;

			// C = AT . B
			cublasSgemm(cbls_handle, //
						CUBLAS_OP_N, CUBLAS_OP_T,
						B.cols, this->cols, this->rows, // B cols, A cols, A rows
						&alpha,
						B.mat, B.cols,		   // B, B cols
						this->mat, this->cols, // A, A cols
						&beta,
						res.mat, B.cols); // C, B cols

			return res;
		}
		else
		{
			std::cerr << "\nError in Tdot! Check arguments";
			return np::ArrayGPU<TP>(1, 1, 0);
		}
	}

	// dot with second mat tranposed
	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::dotT(const ArrayGPU<TP> &B) const
	{
		/*
			C = A @ B.T
			input:
				A: shape MxK
				B: shape NxK
			output:
				C: shape MxN

		*/
		// condition for dot product
		if (this->cols == B.cols)
		{
			ArrayGPU<TP> res(this->rows, B.rows);

			const float alpha = 1.0f;
			const float beta = 0.0f;

			cublasSgemm(cbls_handle, //
						CUBLAS_OP_T, CUBLAS_OP_N,
						B.rows, this->rows, this->cols, // B cols, A rows, A cols
						&alpha,
						B.mat, B.cols,		   // B, B cols
						this->mat, this->cols, // A, A cols
						&beta,
						res.mat, B.rows); // C, B cols

			return res;
		}
		else
		{
			std::cerr << "\nError in dotT ! Check arguments";
			return np::ArrayGPU<TP>(1, 1, 0);
		}
	}

	// assignment operator overload
	template <typename TP>
	void ArrayGPU<TP>::operator=(const ArrayGPU<TP> &A)
	{
		// free the contents
		CUDA_CALL(cudaFree(this->mat));

		// allocate memory
		this->rows = A.rows;
		this->cols = A.cols;
		CUDA_CALL(cudaMalloc((void **)&this->mat, this->rows * this->cols * sizeof(TP)));

		this->copyFromGPU(A.mat);
	}

	// add functions
	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::operator+(const ArrayGPU<TP> &B) const
	{
		if (this->rows == 1 && this->cols == 1)
		{
			// A is scalar
			ArrayGPU<TP> res(B.rows, B.cols);

			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));

			kernelScalarOpMat<TP, 1><<<grid, block>>>(this->at(0), B.mat, res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}
		else if (B.rows == 1 && B.cols == 1)
		{
			// B is scalar
			ArrayGPU<TP> res(this->rows, this->cols);

			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));

			kernelMatOpScalar<TP, 1><<<grid, block>>>(this->mat, B.at(0), res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}
		// if A is vector
		// A vector ki dim, is equal to either col or row of B
		// row vector. will extend along cols if possible. (prioritising in case of square matrix)
		// vice versa for cols

		else if ((this->cols == 1 && this->rows == B.rows) || (this->rows == 1 && this->cols == B.rows))
		{
			// along rows add kr
			ArrayGPU<TP> res(B.rows, B.cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelVecOpMatAlongCols<TP, 1><<<grid, block>>>(this->mat, B.mat, res.mat, res.size(), B.cols);
			cudaDeviceSynchronize();

			return res;
		}
		else if ((this->cols == 1 && this->rows == B.cols) || (this->rows == 1 && this->cols == B.cols))
		{
			// along cols add kr
			ArrayGPU<TP> res(B.rows, B.cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelVecOpMatAlongRows<TP, 1><<<grid, block>>>(this->mat, B.mat, res.mat, res.size(), B.cols);
			cudaDeviceSynchronize();

			return res;
		}
		// B is vetor
		// B vector ki dim, is eq to either col or row of B
		// row vector. will extend along cols if possible. (prioritising in case of square matrix)
		else if ((B.cols == 1 && this->rows == B.rows) || (B.rows == 1 && this->rows == B.cols))
		{
			// along rows add kr
			ArrayGPU<TP> res(this->rows, this->cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatOpVecAlongCols<TP, 1><<<grid, block>>>(this->mat, B.mat, res.mat, res.size(), this->cols);
			cudaDeviceSynchronize();

			return res;
		}
		else if ((B.cols == 1 && this->cols == B.rows) || (B.rows == 1 && this->cols == B.cols))
		{
			// along cols add kr
			ArrayGPU<TP> res(this->rows, this->cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatOpVecAlongRows<TP, 1><<<grid, block>>>(this->mat, B.mat, res.mat, res.size(), this->cols);
			cudaDeviceSynchronize();

			return res;
		}
		else if (this->rows == B.rows && this->cols == B.cols)
		{
			// A and B both are matrices of same dimensions
			ArrayGPU<TP> res(this->rows, this->cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatOpMat<TP, 1><<<grid, block>>>(this->mat, B.mat, res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}
		else
		{
			std::cerr << "\nError in +! Check arguments";
			return np::ArrayGPU<TP>(1, 1, 0);
		}
	}

	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::operator+(const TP Scalar) const
	{
		ArrayGPU<TP> res(this->rows, this->cols);
		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(ceil(res.size(), block.x));
		kernelMatOpScalar<TP, 1><<<grid, block>>>(this->mat, Scalar, res.mat, res.size());
		cudaDeviceSynchronize();
		return res;
	}

	template <typename TP>
	ArrayGPU<TP> operator+(const TP Scal, const ArrayGPU<TP> &A)
	{
		ArrayGPU<TP> res(A.rows, A.cols);

		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(ceil(res.size(), block.x));

		kernelScalarOpMat<TP, 1><<<grid, block>>>(Scal, A.mat, res.mat, res.size());
		cudaDeviceSynchronize();
		return res;
	}

	// subtraction
	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::operator-(const ArrayGPU<TP> &B) const
	{
		if (this->rows == 1 && this->cols == 1)
		{
			// A is scalar
			ArrayGPU<TP> res(B.rows, B.cols);

			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));

			kernelScalarOpMat<TP, 2><<<grid, block>>>(this->at(0), B.mat, res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}
		else if (B.rows == 1 && B.cols == 1)
		{
			// B is scalar
			ArrayGPU<TP> res(this->rows, this->cols);

			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));

			kernelMatOpScalar<TP, 2><<<grid, block>>>(this->mat, B.at(0), res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}
		// if A is vector
		// A vector ki dim, is equal to either col or row of B
		// row vector. will extend along cols if possible. (prioritising in case of square matrix)
		// vice versa for cols

		else if ((this->cols == 1 && this->rows == B.rows) || (this->rows == 1 && this->cols == B.rows))
		{
			// along rows add kr
			ArrayGPU<TP> res(B.rows, B.cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelVecOpMatAlongCols<TP, 2><<<grid, block>>>(this->mat, B.mat, res.mat, res.size(), B.cols);
			cudaDeviceSynchronize();

			return res;
		}
		else if ((this->cols == 1 && this->rows == B.cols) || (this->rows == 1 && this->cols == B.cols))
		{
			// along cols add kr
			ArrayGPU<TP> res(B.rows, B.cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelVecOpMatAlongRows<TP, 2><<<grid, block>>>(this->mat, B.mat, res.mat, res.size(), B.cols);
			cudaDeviceSynchronize();

			return res;
		}
		// B is vetor
		// B vector ki dim, is eq to either col or row of B
		// row vector. will extend along cols if possible. (prioritising in case of square matrix)
		else if ((B.cols == 1 && this->rows == B.rows) || (B.rows == 1 && this->rows == B.cols))
		{
			// along rows add kr
			ArrayGPU<TP> res(this->rows, this->cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatOpVecAlongCols<TP, 2><<<grid, block>>>(this->mat, B.mat, res.mat, res.size(), this->cols);
			cudaDeviceSynchronize();

			return res;
		}
		else if ((B.cols == 1 && this->cols == B.rows) || (B.rows == 1 && this->cols == B.cols))
		{
			// along cols add kr
			ArrayGPU<TP> res(this->rows, this->cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatOpVecAlongRows<TP, 2><<<grid, block>>>(this->mat, B.mat, res.mat, res.size(), this->cols);
			cudaDeviceSynchronize();

			return res;
		}
		else if (this->rows == B.rows && this->cols == B.cols)
		{
			// A and B both are matrices of same dimensions
			ArrayGPU<TP> res(this->rows, this->cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatOpMat<TP, 2><<<grid, block>>>(this->mat, B.mat, res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}
		else
		{
			std::cerr << "\nError in +! Check arguments";
			return np::ArrayGPU<TP>(1, 1, 0);
		}
	}

	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::operator-(const TP Scalar) const
	{
		ArrayGPU<TP> res(this->rows, this->cols);
		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(ceil(res.size(), block.x));
		kernelMatOpScalar<TP, 2><<<grid, block>>>(this->mat, Scalar, res.mat, res.size());
		cudaDeviceSynchronize();
		return res;
	}

	template <typename TP>
	ArrayGPU<TP> operator-(const TP Scal, const ArrayGPU<TP> &A)
	{
		ArrayGPU<TP> res(A.rows, A.cols);

		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(ceil(res.size(), block.x));

		kernelScalarOpMat<TP, 2><<<grid, block>>>(Scal, A.mat, res.mat, res.size());
		cudaDeviceSynchronize();
		return res;
	}

	// multiplication
	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::operator*(const ArrayGPU<TP> &B) const
	{
		if (this->rows == 1 && this->cols == 1)
		{
			// A is scalar
			ArrayGPU<TP> res(B.rows, B.cols);

			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));

			kernelScalarOpMat<TP, 3><<<grid, block>>>(this->at(0), B.mat, res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}
		else if (B.rows == 1 && B.cols == 1)
		{
			// B is scalar
			ArrayGPU<TP> res(this->rows, this->cols);

			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));

			kernelMatOpScalar<TP, 3><<<grid, block>>>(this->mat, B.at(0), res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}
		// if A is vector
		// A vector ki dim, is equal to either col or row of B
		// row vector. will extend along cols if possible. (prioritising in case of square matrix)
		// vice versa for cols

		else if ((this->cols == 1 && this->rows == B.rows) || (this->rows == 1 && this->cols == B.rows))
		{
			// along rows add kr
			ArrayGPU<TP> res(B.rows, B.cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelVecOpMatAlongCols<TP, 3><<<grid, block>>>(this->mat, B.mat, res.mat, res.size(), B.cols);
			cudaDeviceSynchronize();

			return res;
		}
		else if ((this->cols == 1 && this->rows == B.cols) || (this->rows == 1 && this->cols == B.cols))
		{
			// along cols add kr
			ArrayGPU<TP> res(B.rows, B.cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelVecOpMatAlongRows<TP, 3><<<grid, block>>>(this->mat, B.mat, res.mat, res.size(), B.cols);
			cudaDeviceSynchronize();

			return res;
		}
		// B is vetor
		// B vector ki dim, is eq to either col or row of B
		// row vector. will extend along cols if possible. (prioritising in case of square matrix)
		else if ((B.cols == 1 && this->rows == B.rows) || (B.rows == 1 && this->rows == B.cols))
		{
			// along rows add kr
			ArrayGPU<TP> res(this->rows, this->cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatOpVecAlongCols<TP, 3><<<grid, block>>>(this->mat, B.mat, res.mat, res.size(), this->cols);
			cudaDeviceSynchronize();

			return res;
		}
		else if ((B.cols == 1 && this->cols == B.rows) || (B.rows == 1 && this->cols == B.cols))
		{
			// along cols add kr
			ArrayGPU<TP> res(this->rows, this->cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatOpVecAlongRows<TP, 3><<<grid, block>>>(this->mat, B.mat, res.mat, res.size(), this->cols);
			cudaDeviceSynchronize();

			return res;
		}
		else if (this->rows == B.rows && this->cols == B.cols)
		{
			// A and B both are matrices of same dimensions
			ArrayGPU<TP> res(this->rows, this->cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatOpMat<TP, 3><<<grid, block>>>(this->mat, B.mat, res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}
		else
		{
			std::cerr << "\nError in +! Check arguments";
			return np::ArrayGPU<TP>(1, 1, 0);
		}
	}

	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::operator*(const TP Scalar) const
	{
		ArrayGPU<TP> res(this->rows, this->cols);
		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(ceil(res.size(), block.x));
		kernelMatOpScalar<TP, 3><<<grid, block>>>(this->mat, Scalar, res.mat, res.size());
		cudaDeviceSynchronize();
		return res;
	}

	template <typename TP>
	ArrayGPU<TP> operator*(const TP Scal, const ArrayGPU<TP> &A)
	{
		ArrayGPU<TP> res(A.rows, A.cols);

		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(ceil(res.size(), block.x));

		kernelScalarOpMat<TP, 3><<<grid, block>>>(Scal, A.mat, res.mat, res.size());
		cudaDeviceSynchronize();
		return res;
	}

	// division
	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::operator/(const ArrayGPU<TP> &B) const
	{
		if (this->rows == 1 && this->cols == 1)
		{
			// A is scalar
			ArrayGPU<TP> res(B.rows, B.cols);

			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));

			kernelScalarOpMat<TP, 4><<<grid, block>>>(this->at(0), B.mat, res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}
		else if (B.rows == 1 && B.cols == 1)
		{
			// B is scalar
			ArrayGPU<TP> res(this->rows, this->cols);

			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));

			kernelMatOpScalar<TP, 4><<<grid, block>>>(this->mat, B.at(0), res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}
		// if A is vector
		// A vector ki dim, is equal to either col or row of B
		// row vector. will extend along cols if possible. (prioritising in case of square matrix)
		// vice versa for cols

		else if ((this->cols == 1 && this->rows == B.rows) || (this->rows == 1 && this->cols == B.rows))
		{
			// along rows add kr
			ArrayGPU<TP> res(B.rows, B.cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelVecOpMatAlongCols<TP, 4><<<grid, block>>>(this->mat, B.mat, res.mat, res.size(), B.cols);
			cudaDeviceSynchronize();

			return res;
		}
		else if ((this->cols == 1 && this->rows == B.cols) || (this->rows == 1 && this->cols == B.cols))
		{
			// along cols add kr
			ArrayGPU<TP> res(B.rows, B.cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelVecOpMatAlongRows<TP, 4><<<grid, block>>>(this->mat, B.mat, res.mat, res.size(), B.cols);
			cudaDeviceSynchronize();

			return res;
		}
		// B is vetor
		// B vector ki dim, is eq to either col or row of B
		// row vector. will extend along cols if possible. (prioritising in case of square matrix)
		else if ((B.cols == 1 && this->rows == B.rows) || (B.rows == 1 && this->rows == B.cols))
		{
			// along rows add kr
			ArrayGPU<TP> res(this->rows, this->cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatOpVecAlongCols<TP, 4><<<grid, block>>>(this->mat, B.mat, res.mat, res.size(), this->cols);
			cudaDeviceSynchronize();

			return res;
		}
		else if ((B.cols == 1 && this->cols == B.rows) || (B.rows == 1 && this->cols == B.cols))
		{
			// along cols add kr
			ArrayGPU<TP> res(this->rows, this->cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatOpVecAlongRows<TP, 4><<<grid, block>>>(this->mat, B.mat, res.mat, res.size(), this->cols);
			cudaDeviceSynchronize();

			return res;
		}
		else if (this->rows == B.rows && this->cols == B.cols)
		{
			// A and B both are matrices of same dimensions
			ArrayGPU<TP> res(this->rows, this->cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatOpMat<TP, 4><<<grid, block>>>(this->mat, B.mat, res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}
		else
		{
			std::cerr << "\nError in +! Check arguments";
			return np::ArrayGPU<TP>(1, 1, 0);
		}
	}

	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::operator/(const TP Scalar) const
	{
		ArrayGPU<TP> res(this->rows, this->cols);
		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(ceil(res.size(), block.x));
		kernelMatOpScalar<TP, 4><<<grid, block>>>(this->mat, Scalar, res.mat, res.size());
		cudaDeviceSynchronize();
		return res;
	}

	template <typename TP>
	ArrayGPU<TP> operator/(const TP Scal, const ArrayGPU<TP> &A)
	{
		ArrayGPU<TP> res(A.rows, A.cols);

		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(ceil(res.size(), block.x));

		kernelScalarOpMat<TP, 4><<<grid, block>>>(Scal, A.mat, res.mat, res.size());
		cudaDeviceSynchronize();
		return res;
	}

	// unary negation operator
	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::operator-() const
	{
		ArrayGPU<TP> res(this->rows, this->cols);
		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(ceil(res.size(), block.x));
		kernelMatOpScalar<TP, 3><<<grid, block>>>(this->mat, -1, res.mat, res.size());
		cudaDeviceSynchronize();
		return res;
	}

	// returns an array of 0s and 1s depending on true or false of the conditions.
	//  element wise comparison

	// <
	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::operator<(const ArrayGPU<TP> &B) const
	{
		if (this->rows == 1 && this->cols == 1)
		{
			// A is scalar
			ArrayGPU<TP> res(B.rows, B.cols);

			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));

			kernelScalarOpMat<TP, 5><<<grid, block>>>(this->at(0), B.mat, res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}
		else if (B.rows == 1 && B.cols == 1)
		{
			// B is scalar
			ArrayGPU<TP> res(this->rows, this->cols);

			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));

			kernelMatOpScalar<TP, 5><<<grid, block>>>(this->mat, B.at(0), res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}
		// if A is vector
		// A vector ki dim, is equal to either col or row of B
		// row vector. will extend along cols if possible. (prioritising in case of square matrix)
		// vice versa for cols

		else if ((this->cols == 1 && this->rows == B.rows) || (this->rows == 1 && this->cols == B.rows))
		{
			// along rows add kr
			ArrayGPU<TP> res(B.rows, B.cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelVecOpMatAlongCols<TP, 5><<<grid, block>>>(this->mat, B.mat, res.mat, res.size(), B.cols);
			cudaDeviceSynchronize();

			return res;
		}
		else if ((this->cols == 1 && this->rows == B.cols) || (this->rows == 1 && this->cols == B.cols))
		{
			// along cols add kr
			ArrayGPU<TP> res(B.rows, B.cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelVecOpMatAlongRows<TP, 5><<<grid, block>>>(this->mat, B.mat, res.mat, res.size(), B.cols);
			cudaDeviceSynchronize();

			return res;
		}
		// B is vetor
		// B vector ki dim, is eq to either col or row of B
		// row vector. will extend along cols if possible. (prioritising in case of square matrix)
		else if ((B.cols == 1 && this->rows == B.rows) || (B.rows == 1 && this->rows == B.cols))
		{
			// along rows add kr
			ArrayGPU<TP> res(this->rows, this->cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatOpVecAlongCols<TP, 5><<<grid, block>>>(this->mat, B.mat, res.mat, res.size(), this->cols);
			cudaDeviceSynchronize();

			return res;
		}
		else if ((B.cols == 1 && this->cols == B.rows) || (B.rows == 1 && this->cols == B.cols))
		{
			// along cols add kr
			ArrayGPU<TP> res(this->rows, this->cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatOpVecAlongRows<TP, 5><<<grid, block>>>(this->mat, B.mat, res.mat, res.size(), this->cols);
			cudaDeviceSynchronize();

			return res;
		}
		else if (this->rows == B.rows && this->cols == B.cols)
		{
			// A and B both are matrices of same dimensions
			ArrayGPU<TP> res(this->rows, this->cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatOpMat<TP, 5><<<grid, block>>>(this->mat, B.mat, res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}
		else
		{
			std::cerr << "\nError in +! Check arguments";
			return np::ArrayGPU<TP>(1, 1, 0);
		}
	}

	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::operator<(const TP Scalar) const
	{
		ArrayGPU<TP> res(this->rows, this->cols);
		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(ceil(res.size(), block.x));
		kernelMatOpScalar<TP, 5><<<grid, block>>>(this->mat, Scalar, res.mat, res.size());
		cudaDeviceSynchronize();
		return res;
	}

	template <typename TP>
	ArrayGPU<TP> operator<(const TP Scal, const ArrayGPU<TP> &A)
	{
		ArrayGPU<TP> res(A.rows, A.cols);

		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(ceil(res.size(), block.x));

		kernelScalarOpMat<TP, 5><<<grid, block>>>(Scal, A.mat, res.mat, res.size());
		cudaDeviceSynchronize();
		return res;
	}

	// <=
	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::operator<=(const ArrayGPU<TP> &B) const
	{
		if (this->rows == 1 && this->cols == 1)
		{
			// A is scalar
			ArrayGPU<TP> res(B.rows, B.cols);

			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));

			kernelScalarOpMat<TP, 6><<<grid, block>>>(this->at(0), B.mat, res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}
		else if (B.rows == 1 && B.cols == 1)
		{
			// B is scalar
			ArrayGPU<TP> res(this->rows, this->cols);

			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));

			kernelMatOpScalar<TP, 6><<<grid, block>>>(this->mat, B.at(0), res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}
		// if A is vector
		// A vector ki dim, is equal to either col or row of B
		// row vector. will extend along cols if possible. (prioritising in case of square matrix)
		// vice versa for cols

		else if ((this->cols == 1 && this->rows == B.rows) || (this->rows == 1 && this->cols == B.rows))
		{
			// along rows add kr
			ArrayGPU<TP> res(B.rows, B.cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelVecOpMatAlongCols<TP, 6><<<grid, block>>>(this->mat, B.mat, res.mat, res.size(), B.cols);
			cudaDeviceSynchronize();

			return res;
		}
		else if ((this->cols == 1 && this->rows == B.cols) || (this->rows == 1 && this->cols == B.cols))
		{
			// along cols add kr
			ArrayGPU<TP> res(B.rows, B.cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelVecOpMatAlongRows<TP, 6><<<grid, block>>>(this->mat, B.mat, res.mat, res.size(), B.cols);
			cudaDeviceSynchronize();

			return res;
		}
		// B is vetor
		// B vector ki dim, is eq to either col or row of B
		// row vector. will extend along cols if possible. (prioritising in case of square matrix)
		else if ((B.cols == 1 && this->rows == B.rows) || (B.rows == 1 && this->rows == B.cols))
		{
			// along rows add kr
			ArrayGPU<TP> res(this->rows, this->cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatOpVecAlongCols<TP, 6><<<grid, block>>>(this->mat, B.mat, res.mat, res.size(), this->cols);
			cudaDeviceSynchronize();

			return res;
		}
		else if ((B.cols == 1 && this->cols == B.rows) || (B.rows == 1 && this->cols == B.cols))
		{
			// along cols add kr
			ArrayGPU<TP> res(this->rows, this->cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatOpVecAlongRows<TP, 6><<<grid, block>>>(this->mat, B.mat, res.mat, res.size(), this->cols);
			cudaDeviceSynchronize();

			return res;
		}
		else if (this->rows == B.rows && this->cols == B.cols)
		{
			// A and B both are matrices of same dimensions
			ArrayGPU<TP> res(this->rows, this->cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatOpMat<TP, 6><<<grid, block>>>(this->mat, B.mat, res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}
		else
		{
			std::cerr << "\nError in +! Check arguments";
			return np::ArrayGPU<TP>(1, 1, 0);
		}
	}

	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::operator<=(const TP Scalar) const
	{
		ArrayGPU<TP> res(this->rows, this->cols);
		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(ceil(res.size(), block.x));
		kernelMatOpScalar<TP, 6><<<grid, block>>>(this->mat, Scalar, res.mat, res.size());
		cudaDeviceSynchronize();
		return res;
	}

	template <typename TP>
	ArrayGPU<TP> operator<=(const TP Scal, const ArrayGPU<TP> &A)
	{
		ArrayGPU<TP> res(A.rows, A.cols);

		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(ceil(res.size(), block.x));

		kernelScalarOpMat<TP, 6><<<grid, block>>>(Scal, A.mat, res.mat, res.size());
		cudaDeviceSynchronize();
		return res;
	}

	// >
	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::operator>(const ArrayGPU<TP> &B) const
	{
		if (this->rows == 1 && this->cols == 1)
		{
			// A is scalar
			ArrayGPU<TP> res(B.rows, B.cols);

			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));

			kernelScalarOpMat<TP, 7><<<grid, block>>>(this->at(0), B.mat, res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}
		else if (B.rows == 1 && B.cols == 1)
		{
			// B is scalar
			ArrayGPU<TP> res(this->rows, this->cols);

			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));

			kernelMatOpScalar<TP, 7><<<grid, block>>>(this->mat, B.at(0), res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}
		// if A is vector
		// A vector ki dim, is equal to either col or row of B
		// row vector. will extend along cols if possible. (prioritising in case of square matrix)
		// vice versa for cols

		else if ((this->cols == 1 && this->rows == B.rows) || (this->rows == 1 && this->cols == B.rows))
		{
			// along rows add kr
			ArrayGPU<TP> res(B.rows, B.cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelVecOpMatAlongCols<TP, 7><<<grid, block>>>(this->mat, B.mat, res.mat, res.size(), B.cols);
			cudaDeviceSynchronize();

			return res;
		}
		else if ((this->cols == 1 && this->rows == B.cols) || (this->rows == 1 && this->cols == B.cols))
		{
			// along cols add kr
			ArrayGPU<TP> res(B.rows, B.cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelVecOpMatAlongRows<TP, 7><<<grid, block>>>(this->mat, B.mat, res.mat, res.size(), B.cols);
			cudaDeviceSynchronize();

			return res;
		}
		// B is vetor
		// B vector ki dim, is eq to either col or row of B
		// row vector. will extend along cols if possible. (prioritising in case of square matrix)
		else if ((B.cols == 1 && this->rows == B.rows) || (B.rows == 1 && this->rows == B.cols))
		{
			// along rows add kr
			ArrayGPU<TP> res(this->rows, this->cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatOpVecAlongCols<TP, 7><<<grid, block>>>(this->mat, B.mat, res.mat, res.size(), this->cols);
			cudaDeviceSynchronize();

			return res;
		}
		else if ((B.cols == 1 && this->cols == B.rows) || (B.rows == 1 && this->cols == B.cols))
		{
			// along cols add kr
			ArrayGPU<TP> res(this->rows, this->cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatOpVecAlongRows<TP, 7><<<grid, block>>>(this->mat, B.mat, res.mat, res.size(), this->cols);
			cudaDeviceSynchronize();

			return res;
		}
		else if (this->rows == B.rows && this->cols == B.cols)
		{
			// A and B both are matrices of same dimensions
			ArrayGPU<TP> res(this->rows, this->cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatOpMat<TP, 7><<<grid, block>>>(this->mat, B.mat, res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}
		else
		{
			std::cerr << "\nError in +! Check arguments";
			return np::ArrayGPU<TP>(1, 1, 0);
		}
	}

	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::operator>(const TP Scalar) const
	{
		ArrayGPU<TP> res(this->rows, this->cols);
		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(ceil(res.size(), block.x));
		kernelMatOpScalar<TP, 7><<<grid, block>>>(this->mat, Scalar, res.mat, res.size());
		cudaDeviceSynchronize();
		return res;
	}

	template <typename TP>
	ArrayGPU<TP> operator>(const TP Scal, const ArrayGPU<TP> &A)
	{
		ArrayGPU<TP> res(A.rows, A.cols);

		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(ceil(res.size(), block.x));

		kernelScalarOpMat<TP, 7><<<grid, block>>>(Scal, A.mat, res.mat, res.size());
		cudaDeviceSynchronize();
		return res;
	}

	// >=
	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::operator>=(const ArrayGPU<TP> &B) const
	{
		if (this->rows == 1 && this->cols == 1)
		{
			// A is scalar
			ArrayGPU<TP> res(B.rows, B.cols);

			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));

			kernelScalarOpMat<TP, 8><<<grid, block>>>(this->at(0), B.mat, res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}
		else if (B.rows == 1 && B.cols == 1)
		{
			// B is scalar
			ArrayGPU<TP> res(this->rows, this->cols);

			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));

			kernelMatOpScalar<TP, 8><<<grid, block>>>(this->mat, B.at(0), res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}
		// if A is vector
		// A vector ki dim, is equal to either col or row of B
		// row vector. will extend along cols if possible. (prioritising in case of square matrix)
		// vice versa for cols

		else if ((this->cols == 1 && this->rows == B.rows) || (this->rows == 1 && this->cols == B.rows))
		{
			// along rows add kr
			ArrayGPU<TP> res(B.rows, B.cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelVecOpMatAlongCols<TP, 8><<<grid, block>>>(this->mat, B.mat, res.mat, res.size(), B.cols);
			cudaDeviceSynchronize();

			return res;
		}
		else if ((this->cols == 1 && this->rows == B.cols) || (this->rows == 1 && this->cols == B.cols))
		{
			// along cols add kr
			ArrayGPU<TP> res(B.rows, B.cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelVecOpMatAlongRows<TP, 8><<<grid, block>>>(this->mat, B.mat, res.mat, res.size(), B.cols);
			cudaDeviceSynchronize();

			return res;
		}
		// B is vetor
		// B vector ki dim, is eq to either col or row of B
		// row vector. will extend along cols if possible. (prioritising in case of square matrix)
		else if ((B.cols == 1 && this->rows == B.rows) || (B.rows == 1 && this->rows == B.cols))
		{
			// along rows add kr
			ArrayGPU<TP> res(this->rows, this->cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatOpVecAlongCols<TP, 8><<<grid, block>>>(this->mat, B.mat, res.mat, res.size(), this->cols);
			cudaDeviceSynchronize();

			return res;
		}
		else if ((B.cols == 1 && this->cols == B.rows) || (B.rows == 1 && this->cols == B.cols))
		{
			// along cols add kr
			ArrayGPU<TP> res(this->rows, this->cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatOpVecAlongRows<TP, 8><<<grid, block>>>(this->mat, B.mat, res.mat, res.size(), this->cols);
			cudaDeviceSynchronize();

			return res;
		}
		else if (this->rows == B.rows && this->cols == B.cols)
		{
			// A and B both are matrices of same dimensions
			ArrayGPU<TP> res(this->rows, this->cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatOpMat<TP, 8><<<grid, block>>>(this->mat, B.mat, res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}
		else
		{
			std::cerr << "\nError in +! Check arguments";
			return np::ArrayGPU<TP>(1, 1, 0);
		}
	}

	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::operator>=(const TP Scalar) const
	{
		ArrayGPU<TP> res(this->rows, this->cols);
		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(ceil(res.size(), block.x));
		kernelMatOpScalar<TP, 8><<<grid, block>>>(this->mat, Scalar, res.mat, res.size());
		cudaDeviceSynchronize();
		return res;
	}

	template <typename TP>
	ArrayGPU<TP> operator>=(const TP Scal, const ArrayGPU<TP> &A)
	{
		ArrayGPU<TP> res(A.rows, A.cols);

		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(ceil(res.size(), block.x));

		kernelScalarOpMat<TP, 8><<<grid, block>>>(Scal, A.mat, res.mat, res.size());
		cudaDeviceSynchronize();
		return res;
	}

	// ==
	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::operator==(const ArrayGPU<TP> &B) const
	{
		if (this->rows == 1 && this->cols == 1)
		{
			// A is scalar
			ArrayGPU<TP> res(B.rows, B.cols);

			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));

			kernelScalarOpMat<TP, 9><<<grid, block>>>(this->at(0), B.mat, res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}
		else if (B.rows == 1 && B.cols == 1)
		{
			// B is scalar
			ArrayGPU<TP> res(this->rows, this->cols);

			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));

			kernelMatOpScalar<TP, 9><<<grid, block>>>(this->mat, B.at(0), res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}
		// if A is vector
		// A vector ki dim, is equal to either col or row of B
		// row vector. will extend along cols if possible. (prioritising in case of square matrix)
		// vice versa for cols

		else if ((this->cols == 1 && this->rows == B.rows) || (this->rows == 1 && this->cols == B.rows))
		{
			// along rows add kr
			ArrayGPU<TP> res(B.rows, B.cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelVecOpMatAlongCols<TP, 9><<<grid, block>>>(this->mat, B.mat, res.mat, res.size(), B.cols);
			cudaDeviceSynchronize();

			return res;
		}
		else if ((this->cols == 1 && this->rows == B.cols) || (this->rows == 1 && this->cols == B.cols))
		{
			// along cols add kr
			ArrayGPU<TP> res(B.rows, B.cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelVecOpMatAlongRows<TP, 9><<<grid, block>>>(this->mat, B.mat, res.mat, res.size(), B.cols);
			cudaDeviceSynchronize();

			return res;
		}
		// B is vetor
		// B vector ki dim, is eq to either col or row of B
		// row vector. will extend along cols if possible. (prioritising in case of square matrix)
		else if ((B.cols == 1 && this->rows == B.rows) || (B.rows == 1 && this->rows == B.cols))
		{
			// along rows add kr
			ArrayGPU<TP> res(this->rows, this->cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatOpVecAlongCols<TP, 9><<<grid, block>>>(this->mat, B.mat, res.mat, res.size(), this->cols);
			cudaDeviceSynchronize();

			return res;
		}
		else if ((B.cols == 1 && this->cols == B.rows) || (B.rows == 1 && this->cols == B.cols))
		{
			// along cols add kr
			ArrayGPU<TP> res(this->rows, this->cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatOpVecAlongRows<TP, 9><<<grid, block>>>(this->mat, B.mat, res.mat, res.size(), this->cols);
			cudaDeviceSynchronize();

			return res;
		}
		else if (this->rows == B.rows && this->cols == B.cols)
		{
			// A and B both are matrices of same dimensions
			ArrayGPU<TP> res(this->rows, this->cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatOpMat<TP, 9><<<grid, block>>>(this->mat, B.mat, res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}
		else
		{
			std::cerr << "\nError in +! Check arguments";
			return np::ArrayGPU<TP>(1, 1, 0);
		}
	}

	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::operator==(const TP Scalar) const
	{
		ArrayGPU<TP> res(this->rows, this->cols);
		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(ceil(res.size(), block.x));
		kernelMatOpScalar<TP, 9><<<grid, block>>>(this->mat, Scalar, res.mat, res.size());
		cudaDeviceSynchronize();
		return res;
	}

	template <typename TP>
	ArrayGPU<TP> operator==(const TP Scal, const ArrayGPU<TP> &A)
	{
		ArrayGPU<TP> res(A.rows, A.cols);

		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(ceil(res.size(), block.x));

		kernelScalarOpMat<TP, 9><<<grid, block>>>(Scal, A.mat, res.mat, res.size());
		cudaDeviceSynchronize();
		return res;
	}

	// !=
	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::operator!=(const ArrayGPU<TP> &B) const
	{
		if (this->rows == 1 && this->cols == 1)
		{
			// A is scalar
			ArrayGPU<TP> res(B.rows, B.cols);

			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));

			kernelScalarOpMat<TP, 10><<<grid, block>>>(this->at(0), B.mat, res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}
		else if (B.rows == 1 && B.cols == 1)
		{
			// B is scalar
			ArrayGPU<TP> res(this->rows, this->cols);

			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));

			kernelMatOpScalar<TP, 10><<<grid, block>>>(this->mat, B.at(0), res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}
		// if A is vector
		// A vector ki dim, is equal to either col or row of B
		// row vector. will extend along cols if possible. (prioritising in case of square matrix)
		// vice versa for cols

		else if ((this->cols == 1 && this->rows == B.rows) || (this->rows == 1 && this->cols == B.rows))
		{
			// along rows add kr
			ArrayGPU<TP> res(B.rows, B.cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelVecOpMatAlongCols<TP, 10><<<grid, block>>>(this->mat, B.mat, res.mat, res.size(), B.cols);
			cudaDeviceSynchronize();

			return res;
		}
		else if ((this->cols == 1 && this->rows == B.cols) || (this->rows == 1 && this->cols == B.cols))
		{
			// along cols add kr
			ArrayGPU<TP> res(B.rows, B.cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelVecOpMatAlongRows<TP, 10><<<grid, block>>>(this->mat, B.mat, res.mat, res.size(), B.cols);
			cudaDeviceSynchronize();

			return res;
		}
		// B is vetor
		// B vector ki dim, is eq to either col or row of B
		// row vector. will extend along cols if possible. (prioritising in case of square matrix)
		else if ((B.cols == 1 && this->rows == B.rows) || (B.rows == 1 && this->rows == B.cols))
		{
			// along rows add kr
			ArrayGPU<TP> res(this->rows, this->cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatOpVecAlongCols<TP, 10><<<grid, block>>>(this->mat, B.mat, res.mat, res.size(), this->cols);
			cudaDeviceSynchronize();

			return res;
		}
		else if ((B.cols == 1 && this->cols == B.rows) || (B.rows == 1 && this->cols == B.cols))
		{
			// along cols add kr
			ArrayGPU<TP> res(this->rows, this->cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatOpVecAlongRows<TP, 10><<<grid, block>>>(this->mat, B.mat, res.mat, res.size(), this->cols);
			cudaDeviceSynchronize();

			return res;
		}
		else if (this->rows == B.rows && this->cols == B.cols)
		{
			// A and B both are matrices of same dimensions
			ArrayGPU<TP> res(this->rows, this->cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(ceil(res.size(), block.x));
			kernelMatOpMat<TP, 10><<<grid, block>>>(this->mat, B.mat, res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}
		else
		{
			std::cerr << "\nError in +! Check arguments";
			return np::ArrayGPU<TP>(1, 1, 0);
		}
	}

	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::operator!=(const TP Scalar) const
	{
		ArrayGPU<TP> res(this->rows, this->cols);
		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(ceil(res.size(), block.x));
		kernelMatOpScalar<TP, 10><<<grid, block>>>(this->mat, Scalar, res.mat, res.size());
		cudaDeviceSynchronize();
		return res;
	}

	template <typename TP>
	ArrayGPU<TP> operator!=(const TP Scal, const ArrayGPU<TP> &A)
	{
		ArrayGPU<TP> res(A.rows, A.cols);

		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(ceil(res.size(), block.x));

		kernelScalarOpMat<TP, 10><<<grid, block>>>(Scal, A.mat, res.mat, res.size());
		cudaDeviceSynchronize();
		return res;
	}




	// sum. along axis or total
	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::sum(const int axis) const
	{
		if (axis == -1)
		{
			// return total sum
			const int BLOCK_SIZE = ((GPU_NUM_CUDA_CORE == 64) ? 64 : 128) * 2;
			dim3 block(BLOCK_SIZE);
			dim3 grid(std::min<int>(ceil(this->size(), block.x), GPU_NUM_SM * 2));

			ArrayGPU<TP> res(1);

			// device pointer tmp
			TP *tmp_d;
			CUDA_CALL(cudaMalloc((void **)&tmp_d, sizeof(TP) * grid.x));
			switch (GPU_NUM_CUDA_CORE)
			{
			case 64:
				kernelReduceSum<TP, 64 * 2><<<grid, block>>>(this->mat, tmp_d, this->size());
				cudaDeviceSynchronize();
				// please guarantee that BLOCK_SIZE > grid.x. otherwise multiple kernel calls will have to be made.
				kernelReduceSum<TP, 64 * 2><<<1, block>>>(tmp_d, res.mat, grid.x);
				cudaDeviceSynchronize();
				break;
			default:
				kernelReduceSum<TP, 128 * 2><<<grid, block>>>(this->mat, tmp_d, this->size());
				cudaDeviceSynchronize();
				// please guarantee that BLOCK_SIZE > grid.x. otherwise multiple kernel calls will have to be made.
				kernelReduceSum<TP, 128 * 2><<<1, block>>>(tmp_d, res.mat, grid.x);
				cudaDeviceSynchronize();
				break;
			}
			CUDA_CALL(cudaFree(tmp_d));
			return res;
		}
		else if (axis == 0)
		{
			// sum along columns. dimension=numCols
			return this->T().sum(1).T();
		}
		else if (axis == 1)
		{
			// sum along rows. output dim = numRows
			ArrayGPU<TP> res(this->rows);

			const int BLOCK_SIZE = ((GPU_NUM_CUDA_CORE == 64) ? 64 : 128) * 2;
			dim3 block(BLOCK_SIZE);
			dim3 grid(std::min<int>(ceil(this->cols, block.x), GPU_NUM_SM * 2));

			TP *tmp_d;
			CUDA_CALL(cudaMalloc((void **)&tmp_d, sizeof(TP) * this->rows * grid.x));

			switch (GPU_NUM_CUDA_CORE)
			{
			case 64:
				for (int i = 0; i < this->rows; ++i)
				{
					kernelReduceSum<TP, 64 * 2><<<grid, block>>>(this->mat + i * this->cols, tmp_d + i * grid.x, this->cols);
				}

				cudaDeviceSynchronize();

				for (int i = 0; i < this->rows; ++i)
				{
					kernelReduceSum<TP, 64 * 2><<<1, block>>>(tmp_d + i * grid.x, res.mat + i, grid.x);
				}

				cudaDeviceSynchronize();
				break;
			default:
				for (int i = 0; i < this->rows; ++i)
				{
					kernelReduceSum<TP, 128 * 2><<<grid, block>>>(this->mat + i * this->cols, tmp_d + i * grid.x, this->cols);
				}

				cudaDeviceSynchronize();

				for (int i = 0; i < this->rows; ++i)
				{
					kernelReduceSum<TP, 128 * 2><<<1, block>>>(tmp_d + i * grid.x, res.mat + i, grid.x);
				}

				cudaDeviceSynchronize();
			}
			return res;
		}
		else
		{
			std::cerr << "\nError in sum! Check arguments";
			return np::ArrayGPU<TP>(1, 1, 0);
		}
	}

	// max. along axis or total
	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::max(const int axis) const
	{
		if (axis == -1)
		{
			// return total sum
			const int BLOCK_SIZE = ((GPU_NUM_CUDA_CORE == 64) ? 64 : 128) * 2;
			dim3 block(BLOCK_SIZE);
			dim3 grid(std::min<int>(ceil(this->size(), block.x), GPU_NUM_SM * 2));
			ArrayGPU<TP> res(1);
			// device pointer tmp
			TP *tmp_d;
			CUDA_CALL(cudaMalloc((void **)&tmp_d, sizeof(TP) * grid.x));
			switch (GPU_NUM_CUDA_CORE)
			{
			case 64:
				kernelReduceMax<TP, 64 * 2><<<grid, block>>>(this->mat, tmp_d, this->size());
				cudaDeviceSynchronize();
				// please guarantee that BLOCK_SIZE > grid.x. otherwise multiple kernel calls will have to be made.
				kernelReduceMax<TP, 64 * 2><<<1, block>>>(tmp_d, res.mat, grid.x);
				cudaDeviceSynchronize();
				break;
			default:
				kernelReduceMax<TP, 128 * 2><<<grid, block>>>(this->mat, tmp_d, this->size());
				cudaDeviceSynchronize();
				// please guarantee that BLOCK_SIZE > grid.x. otherwise multiple kernel calls will have to be made.
				kernelReduceMax<TP, 128 * 2><<<1, block>>>(tmp_d, res.mat, grid.x);
				cudaDeviceSynchronize();
				break;
			}

			CUDA_CALL(cudaFree(tmp_d));

			return res;
		}
		else if (axis == 0)
		{
			// sum along columns. dimension=numCols
			return this->T().max(1).T();
		}
		else if (axis == 1)
		{
			// sum along rows. output dim = numRows
			ArrayGPU<TP> res(this->rows);

			const int BLOCK_SIZE = ((GPU_NUM_CUDA_CORE == 64) ? 64 : 128) * 2;
			dim3 block(BLOCK_SIZE);
			dim3 grid(std::min<int>(ceil(this->cols, block.x), GPU_NUM_SM * 2));

			TP *tmp_d;
			CUDA_CALL(cudaMalloc((void **)&tmp_d, sizeof(TP) * this->rows * grid.x));
			switch (GPU_NUM_CUDA_CORE)
			{
			case 64:
				for (int i = 0; i < this->rows; ++i)
				{
					kernelReduceMax<TP, 64 * 2><<<grid, block>>>(this->mat + i * this->cols, tmp_d + i * grid.x, this->cols);
				}

				cudaDeviceSynchronize();

				for (int i = 0; i < this->rows; ++i)
				{
					kernelReduceMax<TP, 64 * 2><<<1, block>>>(tmp_d + i * grid.x, res.mat + i, grid.x);
				}

				cudaDeviceSynchronize();
				break;
			default:
				for (int i = 0; i < this->rows; ++i)
				{
					kernelReduceMax<TP, 128 * 2><<<grid, block>>>(this->mat + i * this->cols, tmp_d + i * grid.x, this->cols);
				}

				cudaDeviceSynchronize();

				for (int i = 0; i < this->rows; ++i)
				{
					kernelReduceMax<TP, 128 * 2><<<1, block>>>(tmp_d + i * grid.x, res.mat + i, grid.x);
				}

				cudaDeviceSynchronize();
			}
			return res;
		}
		else
		{
			std::cerr << "\nError in max! Check arguments";
			return np::ArrayGPU<TP>(1, 1, 0);
		}
	}

	// min. along axis or total
	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::min(const int axis) const
	{
		if (axis == -1)
		{
			// return total sum
			const int BLOCK_SIZE = ((GPU_NUM_CUDA_CORE == 64) ? 64 : 128) * 2;
			dim3 block(BLOCK_SIZE);
			dim3 grid(std::min<int>(ceil(this->size(), block.x), GPU_NUM_SM * 2));

			ArrayGPU<TP> res(1, 1);
			// device pointer tmp
			TP *tmp_d;
			CUDA_CALL(cudaMalloc((void **)&tmp_d, sizeof(TP) * grid.x));
			switch (GPU_NUM_CUDA_CORE)
			{
			case 64:
				kernelReduceMin<TP, 64 * 2><<<grid, block>>>(this->mat, tmp_d, this->size());
				cudaDeviceSynchronize();
				// please guarantee that BLOCK_SIZE > grid.x. otherwise multiple kernel calls will have to be made.
				kernelReduceMin<TP, 64 * 2><<<1, block>>>(tmp_d, res.mat, grid.x);
				cudaDeviceSynchronize();
				break;
			default:
				kernelReduceMin<TP, 128 * 2><<<grid, block>>>(this->mat, tmp_d, this->size());
				cudaDeviceSynchronize();
				// please guarantee that BLOCK_SIZE > grid.x. otherwise multiple kernel calls will have to be made.
				kernelReduceMin<TP, 128 * 2><<<1, block>>>(tmp_d, res.mat, grid.x);
				cudaDeviceSynchronize();
				break;
			}

			CUDA_CALL(cudaFree(tmp_d));

			return res;
		}
		else if (axis == 0)
		{
			// sum along columns. dimension=numCols
			return this->T().min(1).T();
		}
		else if (axis == 1)
		{
			// sum along rows. output dim = numRows
			ArrayGPU<TP> res(this->rows);

			const int BLOCK_SIZE = ((GPU_NUM_CUDA_CORE == 64) ? 64 : 128) * 2;
			dim3 block(BLOCK_SIZE);
			dim3 grid(std::min<int>(ceil(this->cols, block.x), GPU_NUM_SM * 2));

			TP *tmp_d;
			CUDA_CALL(cudaMalloc((void **)&tmp_d, sizeof(TP) * this->rows * grid.x));
			switch (GPU_NUM_CUDA_CORE)
			{
			case 64:
				for (int i = 0; i < this->rows; ++i)
				{
					kernelReduceMin<TP, 64 * 2><<<grid, block>>>(this->mat + i * this->cols, tmp_d + i * grid.x, this->cols);
				}

				cudaDeviceSynchronize();

				for (int i = 0; i < this->rows; ++i)
				{
					kernelReduceMin<TP, 64 * 2><<<1, block>>>(tmp_d + i * grid.x, res.mat + i, grid.x);
				}

				cudaDeviceSynchronize();
				break;
			default:
				for (int i = 0; i < this->rows; ++i)
				{
					kernelReduceMin<TP, 128 * 2><<<grid, block>>>(this->mat + i * this->cols, tmp_d + i * grid.x, this->cols);
				}

				cudaDeviceSynchronize();

				for (int i = 0; i < this->rows; ++i)
				{
					kernelReduceMin<TP, 128 * 2><<<1, block>>>(tmp_d + i * grid.x, res.mat + i, grid.x);
				}

				cudaDeviceSynchronize();
			}
			return res;
		}
		else
		{
			std::cerr << "\nError in min! Check arguments";
			return np::ArrayGPU<TP>(1, 1, 0);
		}
	}

	// argmax
	template <typename TP>
	ArrayGPU<int> ArrayGPU<TP>::argmax(const int axis) const
	{
		if (axis == -1)
		{
			// return total sum
			const int BLOCK_SIZE = ((GPU_NUM_CUDA_CORE == 64) ? 64 : 128) * 2;
			dim3 block(BLOCK_SIZE);
			dim3 grid(std::min<int>(ceil(this->size(), block.x), GPU_NUM_SM * 2));

			ArrayGPU<TP> res(1);
			ArrayGPU<int> resIdx(1);
			// device pointer tmp
			TP *tmp_A_d;
			CUDA_CALL(cudaMalloc((void **)&tmp_A_d, sizeof(TP) * grid.x));
			int *tmp_A_Idx_d;
			CUDA_CALL(cudaMalloc((void **)&tmp_A_Idx_d, sizeof(const int) * grid.x));

			switch (GPU_NUM_CUDA_CORE)
			{
			case 64:
				kernelReduceArgMax<TP, 64 * 2><<<grid, block>>>(this->mat, tmp_A_d, tmp_A_Idx_d, this->size());
				cudaDeviceSynchronize();
				// please guarantee that BLOCK_SIZE > grid.x. otherwise multiple kernel calls will have to be made.
				kernelReduceArgMax<TP, 64 * 2><<<1, block>>>(tmp_A_d, tmp_A_Idx_d, res.mat, resIdx.mat, grid.x);
				cudaDeviceSynchronize();
				break;
			default:
				kernelReduceArgMax<TP, 128 * 2><<<grid, block>>>(this->mat, tmp_A_d, tmp_A_Idx_d, this->size());
				cudaDeviceSynchronize();
				// please guarantee that BLOCK_SIZE > grid.x. otherwise multiple kernel calls will have to be made.
				kernelReduceArgMax<TP, 128 * 2><<<1, block>>>(tmp_A_d, tmp_A_Idx_d, res.mat, resIdx.mat, grid.x);
				cudaDeviceSynchronize();
			}

			CUDA_CALL(cudaFree(tmp_A_d));

			return resIdx;
		}
		else if (axis == 0)
		{
			// sum along columns. dimension=numCols
			return this->T().argmax(1).T();
		}
		else if (axis == 1)
		{
			// sum along rows. output dim = numRows
			ArrayGPU<TP> res(this->rows);
			ArrayGPU<int> resIdx(this->rows);

			const int BLOCK_SIZE = ((GPU_NUM_CUDA_CORE == 64) ? 64 : 128) * 2;
			dim3 block(BLOCK_SIZE);
			dim3 grid(std::min<int>(ceil(this->cols, block.x), GPU_NUM_SM * 2));

			TP *tmp_A_d;
			CUDA_CALL(cudaMalloc((void **)&tmp_A_d, sizeof(TP) * this->rows * grid.x));
			int *tmp_A_Idx_d;
			CUDA_CALL(cudaMalloc((void **)&tmp_A_Idx_d, sizeof(const int) * this->rows * grid.x));

			switch (GPU_NUM_CUDA_CORE)
			{
			case 64:
				for (int i = 0; i < this->rows; ++i)
				{
					kernelReduceArgMax<TP, 64 * 2><<<grid, block>>>(this->mat + i * this->cols, tmp_A_d + i * grid.x, tmp_A_Idx_d + i * grid.x, this->cols);
				}

				cudaDeviceSynchronize();

				for (int i = 0; i < this->rows; ++i)
				{
					kernelReduceArgMax<TP, 64 * 2><<<1, block>>>(tmp_A_d + i * grid.x, tmp_A_Idx_d + i * grid.x, res.mat + i, resIdx.mat + i, grid.x);
				}

				cudaDeviceSynchronize();
				break;
			default:
				for (int i = 0; i < this->rows; ++i)
				{
					kernelReduceArgMax<TP, 128 * 2><<<grid, block>>>(this->mat + i * this->cols, tmp_A_d + i * grid.x, tmp_A_Idx_d + i * grid.x, this->cols);
				}

				cudaDeviceSynchronize();

				for (int i = 0; i < this->rows; ++i)
				{
					kernelReduceArgMax<TP, 128 * 2><<<1, block>>>(tmp_A_d + i * grid.x, tmp_A_Idx_d + i * grid.x, res.mat + i, resIdx.mat + i, grid.x);
				}

				cudaDeviceSynchronize();
			}

			return resIdx;
		}
		else
		{
			std::cerr << "\nError in argmax! Check arguments";
			return np::ArrayGPU<int>(1, 1, 0);
		}
	}

	// argmin
	// min along axis or total
	template <typename TP>
	ArrayGPU<int> ArrayGPU<TP>::argmin(const int axis) const
	{
		if (axis == -1)
		{
			// return total sum
			const int BLOCK_SIZE = ((GPU_NUM_CUDA_CORE == 64) ? 64 : 128) * 2;
			dim3 block(BLOCK_SIZE);
			dim3 grid(std::min<int>(ceil(this->size(), block.x), GPU_NUM_SM * 2));
			ArrayGPU<TP> res(1);
			ArrayGPU<int> resIdx(1);
			// device pointer tmp
			TP *tmp_A_d;
			CUDA_CALL(cudaMalloc((void **)&tmp_A_d, sizeof(TP) * grid.x));
			int *tmp_A_Idx_d;
			CUDA_CALL(cudaMalloc((void **)&tmp_A_Idx_d, sizeof(const int) * grid.x));

			switch (GPU_NUM_CUDA_CORE)
			{
			case 64:
				kernelReduceArgMin<TP, 64 * 2><<<grid, block>>>(this->mat, tmp_A_d, tmp_A_Idx_d, this->size());
				cudaDeviceSynchronize();
				// please guarantee that BLOCK_SIZE > grid.x. otherwise multiple kernel calls will have to be made.
				kernelReduceArgMin<TP, 64 * 2><<<1, block>>>(tmp_A_d, tmp_A_Idx_d, res.mat, resIdx.mat, grid.x);
				cudaDeviceSynchronize();
				break;
			default:
				kernelReduceArgMin<TP, 128 * 2><<<grid, block>>>(this->mat, tmp_A_d, tmp_A_Idx_d, this->size());
				cudaDeviceSynchronize();
				// please guarantee that BLOCK_SIZE > grid.x. otherwise multiple kernel calls will have to be made.
				kernelReduceArgMin<TP, 128 * 2><<<1, block>>>(tmp_A_d, tmp_A_Idx_d, res.mat, resIdx.mat, grid.x);
				cudaDeviceSynchronize();
			}

			CUDA_CALL(cudaFree(tmp_A_d));

			return resIdx;
		}
		else if (axis == 0)
		{
			// sum along columns. dimension=numCols
			return this->T().argmin(1).T();
		}
		else if (axis == 1)
		{
			// sum along rows. output dim = numRows
			ArrayGPU<TP> res(this->rows);
			ArrayGPU<int> resIdx(this->rows);

			const int BLOCK_SIZE = ((GPU_NUM_CUDA_CORE == 64) ? 64 : 128) * 2;
			dim3 block(BLOCK_SIZE);
			dim3 grid(std::min<int>(ceil(this->cols, block.x), GPU_NUM_SM * 2));

			TP *tmp_A_d;
			CUDA_CALL(cudaMalloc((void **)&tmp_A_d, sizeof(TP) * this->rows * grid.x));
			int *tmp_A_Idx_d;
			CUDA_CALL(cudaMalloc((void **)&tmp_A_Idx_d, sizeof(const int) * this->rows * grid.x));

			switch (GPU_NUM_CUDA_CORE)
			{
			case 64:
				for (int i = 0; i < this->rows; ++i)
				{
					kernelReduceArgMin<TP, 64 * 2><<<grid, block>>>(this->mat + i * this->cols, tmp_A_d + i * grid.x, tmp_A_Idx_d + i * grid.x, this->cols);
				}

				cudaDeviceSynchronize();

				for (int i = 0; i < this->rows; ++i)
				{
					kernelReduceArgMin<TP, 64 * 2><<<1, block>>>(tmp_A_d + i * grid.x, tmp_A_Idx_d + i * grid.x, res.mat + i, resIdx.mat + i, grid.x);
				}

				cudaDeviceSynchronize();
				break;
			default:
				for (int i = 0; i < this->rows; ++i)
				{
					kernelReduceArgMin<TP, 128 * 2><<<grid, block>>>(this->mat + i * this->cols, tmp_A_d + i * grid.x, tmp_A_Idx_d + i * grid.x, this->cols);
				}

				cudaDeviceSynchronize();

				for (int i = 0; i < this->rows; ++i)
				{
					kernelReduceArgMin<TP, 128 * 2><<<1, block>>>(tmp_A_d + i * grid.x, tmp_A_Idx_d + i * grid.x, res.mat + i, resIdx.mat + i, grid.x);
				}

				cudaDeviceSynchronize();
			}

			return resIdx;
		}
		else
		{
			std::cerr << "\nError in argmin! Check arguments.";
			return np::ArrayGPU<int>(1, 1, 0);
		}
	}

	template <typename TP>
	ArrayGPU<TP>::~ArrayGPU()
	{
		cudaFree(this->mat);
	}
}

#endif