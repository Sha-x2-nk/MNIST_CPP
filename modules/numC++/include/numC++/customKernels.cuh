#ifndef CUSTOMKERNELS_CUH
#define CUSTOMKERNELS_CUH

#include <numC++/utils.cuh> // for Operator types

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cmath>

#include <stdio.h>	   // printf
#include <type_traits> // for std::is_same

// curand kernels
// for uniform distribution
template <typename TP>
__global__ void kernelInitializeRandomUnif(TP *arr, const int size, const unsigned long long seed);

// for uniform distribution range = [lo, hi]
template <typename TP>
__global__ void kernelInitializeRandomUnif(TP *arr, const int size, const int lo, const int hi, const unsigned long long seed);

// for normal distribution
template <typename TP>
__global__ void kernelInitializeRandomNorm(TP *arr, const int size, const unsigned long long seed);

// 1 x 1 grid to print matrices
template <typename TP>
__global__ void kernelPrintMat(const TP *in, const int M, const int N);

// kernel to transpose an array.
template <typename TP, int TILE_DIM, int BLOCK_ROWS>
// TILE_WIDTH = 32. 1 block copies 32 x 32 elements.
__global__ void kernelTransposeInMem(const TP *in, TP *out, const int M, const int N);

// kernel to initialise all values of array to val.
template <typename TP>
__global__ void kernelInitMatBroadcast(TP *in, const TP Val, const int size);

// kernel to initialised arange array -> [0 to range), spaced by 1
template <typename TP>
__global__ void kernelInitMatArange(TP *in, const int range);

// kernel to get values at a range of indexes.
template <typename TP>
__global__ void kernelGetMatValues(const TP *in, TP *out, const int *idxs, const int size);

// kernel to get values at a range of indexes.
template <typename TP>
__global__ void kernelGetMatValues(const TP *in, const int rdin, TP *out, const int *rows, const int *cols, const int size);

// sets A[i] to A[i] OP B, for i in idxs
template <typename TP, np::Operation OP>
__global__ void kernelSetMat(TP *A, const TP B, const int *idxs, const int size);

// sets A[i] to A[i] OP B[i], for i in idxs
template <typename TP, np::Operation OP>
__global__ void kernelSetMat(TP *A, const TP *B, const int *idxs, const int size);

// sets A[r][c] to A[r][c] OP B, for r, c in zip(r_idxs, c_idxs)
template <typename TP, np::Operation OP>
__global__ void kernelSetMat(TP *A, const int rdin, const TP B, const int *r_idxs, const int *c_idxs, const int size);

// sets A[r][c] to A[r][c] OP B[r][c], for r, c in zip(r_idxs, c_idxs)
template <typename TP, np::Operation OP>
__global__ void kernelSetMat(TP *A, const int rdin, const TP *B, const int *r_idxs, const int *c_idxs, const int size);

// operator functions

// perform operator on corrosponding elements of 2 matrices
//  C = A op B
template <typename TP, np::Operation OP>
__global__ void kernelMatOpMat(const TP *A, const TP *B, TP *C, const int size);
template <typename TP, np::Operation OP>
__global__ void kernelScalarOpMat(const TP Scal, const TP *A, TP *C, const int size);
// operator on matrix and a scalar.
//  C = A op Scal (broadcasting)
// op scalar to all values of the matrix
template <typename TP, np::Operation OP>
__global__ void kernelMatOpScalar(const TP *A, const TP Scal, TP *C, const int size);
template <typename TP, np::Operation OP>
__global__ void kernelScalarOpMat(const TP Scal, const TP *A, TP *C, const int size);

// op on matrix and vector, mat.rows = vec.dim
// C = A op V (broadcasting)
//  shapeA = M x N matrix
template <typename TP, np::Operation OP>
__global__ void kernelMatOpVecAlongCols(const TP *A, const TP *V, TP *C, const int size, const int N);
template <typename TP, np::Operation OP>
__global__ void kernelVecOpMatAlongCols(const TP *V, const TP *A, TP *C, const int size, const int N);

// operator on matrix and vector, mat.cols = vec.dim
// C = A op V (broadcasting)
// shapeA = M x N matrix
template <typename TP, np::Operation OP>
__global__ void kernelMatOpVecAlongRows(const TP *A, const TP *V, TP *C, const int size, const int N);
template <typename TP, np::Operation OP>
__global__ void kernelVecOpMatAlongRows(const TP *V, const TP *A, TP *C, const int size, const int N);

// maxmin
// compare 2 matrix ( element wise ) and put max / min value in result matrix.
// A = MxN
// B = MxN
// Ci = max(Ai, Bi). (elementwise)
template <typename TP, np::Operation OP>
__global__ void kernelMatMaxminMat(const TP *A, const TP *B, TP *C, const int size);

template <typename TP, np::Operation OP>
__global__ void kernelMatMaxminScalar(const TP *A, const TP B, TP *C, const int size);

template <typename TP, np::Operation OP>
__global__ void kernelMatMaxminVecAlongCols(const TP *A, const TP *V, TP *C, const int size, const int N);

template <typename TP, np::Operation OP>
__global__ void kernelMatMaxminVecAlongRows(const TP *A, const TP *V, TP *C, const int size, const int N);

// npfunctions
// functions per element
// A = MxN
// C = MxN
// Ci = F(Ai)
template <typename TP, np::Operation OP>
__global__ void kernelFMat(const TP *A, TP *C, const int size);

// np.pow
// A = MxN
// C = MxN
// Ci = square(Ai)
template <typename TP>
__global__ void kernelPowMat(const TP *A, const float power, TP *C, const int size);

// REDUCTION
// warp unroll
template <typename TP, np::Operation OP>
__device__ void kernelWarpReduceF(volatile TP *s_A, const int tid);

template <typename TP, int BLOCK_SIZE, np::Operation OP>
__global__ void kernelReduceF(const TP *A, TP *output, const int size);

// warp unroll
template <typename TP, np::Operation OP>
__device__ void kernelWarpReduceArgF(volatile TP *s_A, volatile int *s_Idx, const int tid);

template <typename TP, int BLOCK_SIZE, np::Operation OP>
__global__ void kernelReduceArgF(const TP *A, TP *outputMax, int *outputIdx, const int size);

// second reduction k time p -> idx serial nhi h, to ek idx ka bhi array dena hoga
template <typename TP, int BLOCK_SIZE, np::Operation OP>
__global__ void kernelReduceArgF(const TP *A, const int *A_idx, TP *outputMax, int *outputIdx, const int size);

// np.shuffle
template <typename TP, int BLOCK_SIZE>
__global__ void kernelMatShuffle(TP *A, const int size);

// ########## FUNCTION DEFINITIONS

// for uniform distribution
template <typename TP>
__global__ void kernelInitializeRandomUnif(TP *arr, const int size, int lo, int hi, const unsigned long long seed)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	idx *= 50;
	if (idx < size)
	{
		curandState state;
		curand_init(seed, idx, 0, &state); // Initialize curand state for each thread
		arr[idx] = curand_uniform(&state);  // Generate a random value
		
		for(int i= 1; i< 50 && idx + i< size; ++i){
			arr[idx + i] = ( curand_uniform(&state) * (hi - lo) ) + lo; // generate random value
		}
	}
}

// for normal distribution
template <typename TP>
__global__ void kernelInitializeRandomNorm(TP *arr, const int size, const unsigned long long seed)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	idx *= 50;
	if (idx < size)
	{
		curandState state;
		curand_init(seed, idx, 0, &state); // Initialize curand state for each thread
		arr[idx] = curand_normal(&state);  // Generate a random value
		for(int i= 1; i < 50 && idx + i< size; ++i){
			arr[idx + i] = curand_normal(&state); // generate random value
		}
	}
}

// 1 x 1 grid to print matrices
template <typename TP>
__global__ void kernelPrintMat(const TP *in, const int M, const int N)
{
	// 1, 1 grid. only to print matrix
	for (int r = 0; r < M; ++r)
	{
		for (int c = 0; c < N; ++c)
		{
			if constexpr (std::is_same<TP, int>::value)
			{
				printf("%d ", in[r * N + c]);
			}
			else if constexpr (std::is_same<TP, float>::value)
			{
				printf("%f ", in[r * N + c]);
			}
			else if constexpr (std::is_same<TP, double>::value)
			{
				printf("%lf ", in[r * N + c]);
			}
			else if constexpr (std::is_same<TP, char>::value)
			{
				printf("%c ", in[r * N + c]);
			}
			else
			{
				// Handle other types here
				printf("Unsupported type in kernelPrintMat");
			}
		}
		printf("\n");
	}
}

// kernel to transpose an array.
template <typename TP, int TILE_DIM, int BLOCK_ROWS>
// TILE_WIDTH = 32. 32 x 32 copy krega ek matrix
__global__ void kernelTransposeInMem(const TP *in, TP *out, const int M, const int N)
{
	__shared__ TP tile[TILE_DIM][TILE_DIM + 1];
	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
	{
		if (y + j < M && x < N)
		{
			tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * N + x];
		}
	}

	__syncthreads();

	x = blockIdx.y * TILE_DIM + threadIdx.x; // transpose block offset
	y = blockIdx.x * TILE_DIM + threadIdx.y;

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
	{
		if (y + j < N && x < M)
		{
			out[(y + j) * M + x] = tile[threadIdx.x][threadIdx.y + j];
		}
	}
}

// kernel to initialise all values of array to val.
template <typename TP>
__global__ void kernelInitMatBroadcast(TP *in, const TP Val, const int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int grid_size = blockDim.x * gridDim.x;

	while(idx < size){
		in[idx] = Val;
		idx += grid_size;
	}
}

// kernel to initialised arange array -> [0, range), spaced by 1
template <typename TP>
__global__ void kernelInitMatArange(TP *in, const int range)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int grid_size = blockDim.x * gridDim.x;
	while (idx < range)
	{
		in[idx] = idx;
		idx += grid_size;
	}
}

// get values at idxs in A, store C
template <typename TP>
__global__ void kernelGetMat(const TP *A, TP* C, const int *idxs, const int size){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int grid_size = blockDim.x * gridDim.x;
	
	while (idx < size){
		C[idx] = A[idxs[idx]];
		idx += grid_size;
	}
}

// get values at idxs in A, store C
template <typename TP>
__global__ void kernelGetMat(const TP *A, const int rdin, TP* C, const int *r_idxs, const int *c_idxs, const int size){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int grid_size = blockDim.x * gridDim.x;
	
	while (idx < size){
		C[idx] = A[r_idxs[idx] * rdin + c_idxs[idx]];
		idx += grid_size;
	}
}

// sets A[i] to A[i] OP B, for i in idxs
template <typename TP, np::Operation OP>
__global__ void kernelSetMat(TP *A, const TP B, const int *idxs, const int size){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int grid_size = blockDim.x * gridDim.x;

	while(idx < size){
		if constexpr (OP == np::NP_OP_ADD)
			A[idxs[idx]] += B;
		else if constexpr (OP == np::NP_OP_SUB)
			A[idxs[idx]] -= B;
		else if constexpr (OP == np::NP_OP_MUL)
			A[idxs[idx]] *= B;
		else if constexpr (OP == np::NP_OP_DIV)
			A[idxs[idx]] /= B;
		else if constexpr (OP == np::NP_OP_LESS_THAN)
			A[idxs[idx]] = A[idxs[idx]] < B;
		else if constexpr (OP == np::NP_OP_LESS_THAN_EQ)
			A[idxs[idx]] = A[idxs[idx]] <= B;
		else if constexpr (OP == np::NP_OP_GREATER_THAN)
			A[idxs[idx]] = A[idxs[idx]] > B;
		else if constexpr (OP == np::NP_OP_GREATER_THAN_EQ)
			A[idxs[idx]] = A[idxs[idx]] >= B;
		else if constexpr (OP == np::NP_OP_EQ_EQ)
			A[idxs[idx]] = A[idxs[idx]] == B;
		else if constexpr (OP == np::NP_OP_NOT_EQ)
			A[idxs[idx]] = A[idxs[idx]] != B;
		else if constexpr (OP == np::NP_OP_MINIMUM)
			A[idxs[idx]] = (A[idxs[idx]] < B) ? A[idxs[idx]] : B;
		else if constexpr (OP == np::NP_OP_MAXIMUM)
			A[idxs[idx]] = (A[idxs[idx]] > B) ? A[idxs[idx]] : B;
		else if constexpr (OP == np::NP_OP_EQ)
			A[idxs[idx]] = B;
		else if constexpr (OP == np::NP_F_EXP)
			A[idxs[idx]] = expf(A[idxs[idx]]);
		else if constexpr (OP == np::NP_F_LOG)
			A[idxs[idx]] = logf(A[idxs[idx]]);
		else if constexpr (OP == np::NP_F_SQUARE)
			A[idxs[idx]] *= A[idxs[idx]];
		else if constexpr (OP == np::NP_F_SQRT)
		{
			if constexpr (std::is_same<TP, int>::value)
				A[idxs[idx]] = static_cast<int>(sqrtf(A[idxs[idx]]));
			else if constexpr (std::is_same<TP, float>::value)
				A[idxs[idx]] = sqrtf(A[idxs[idx]]);
			else if constexpr (std::is_same<TP, double>::value)
				A[idxs[idx]] = sqrt(A[idxs[idx]]);
			else
				// Handle other types here
				printf("Unsupported type in kernelSetMat, where F = 22(sqrt)");
		}
		else if constexpr (OP == np::NP_F_POW){
			if constexpr (std::is_same<TP, int>::value)
				A[idxs[idx]] = static_cast<int>(powf(A[idxs[idx]], B));
			else if constexpr (std::is_same<TP, float>::value)
				A[idxs[idx]] = powf(A[idxs[idx]], B);
			else if constexpr (std::is_same<TP, double>::value)
				A[idxs[idx]] = pow(A[idxs[idx]], static_cast<double>(B));
			else
				// Handle other types here
				printf("Unsupported type in kernelSetMat F=23(power)");
		}
		else
			printf("ERROR! INVALID OPERATOR/FUNCTION IN kernelSet.\n");
		idx += grid_size;	
	}
}

// sets A[i] to A[i] OP B[i], for i in idxs
template <typename TP, np::Operation OP>
__global__ void kernelSetMat(TP *A, const TP *B, const int *idxs, const int size){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int grid_size = blockDim.x * gridDim.x;

	while(idx < size){
		if constexpr (OP == np::NP_OP_ADD)
			A[idxs[idx]] += B[idx];
		else if constexpr (OP == np::NP_OP_SUB)
			A[idxs[idx]] -= B[idx];
		else if constexpr (OP == np::NP_OP_MUL)
			A[idxs[idx]] *= B[idx];
		else if constexpr (OP == np::NP_OP_DIV)
			A[idxs[idx]] /= B[idx];
		else if constexpr (OP == np::NP_OP_LESS_THAN)
			A[idxs[idx]] = A[idxs[idx]] < B[idx];
		else if constexpr (OP == np::NP_OP_LESS_THAN_EQ)
			A[idxs[idx]] = A[idxs[idx]] <= B[idx];
		else if constexpr (OP == np::NP_OP_GREATER_THAN)
			A[idxs[idx]] = A[idxs[idx]] > B[idx];
		else if constexpr (OP == np::NP_OP_GREATER_THAN_EQ)
			A[idxs[idx]] = A[idxs[idx]] >= B[idx];
		else if constexpr (OP == np::NP_OP_EQ_EQ)
			A[idxs[idx]] = A[idxs[idx]] == B[idx];
		else if constexpr (OP == np::NP_OP_NOT_EQ)
			A[idxs[idx]] = A[idxs[idx]] != B[idx];
		else if constexpr (OP == np::NP_OP_MINIMUM)
			A[idxs[idx]] = (A[idxs[idx]] < B[idx]) ? A[idxs[idx]] : B[idx];
		else if constexpr (OP == np::NP_OP_MAXIMUM)
			A[idxs[idx]] = (A[idxs[idx]] > B[idx]) ? A[idxs[idx]] : B[idx];
		else if constexpr (OP == np::NP_OP_EQ)
			A[idxs[idx]] = B[idx];
		else if constexpr (OP == np::NP_F_EXP)
			A[idxs[idx]] = expf(A[idxs[idx]]);
		else if constexpr (OP == np::NP_F_LOG)
			A[idxs[idx]] = logf(A[idxs[idx]]);
		else if constexpr (OP == np::NP_F_SQUARE)
			A[idxs[idx]] *= A[idxs[idx]];
		else if constexpr (OP == np::NP_F_SQRT)
		{
			if constexpr (std::is_same<TP, int>::value)
				A[idxs[idx]] = static_cast<int>(sqrtf(A[idxs[idx]]));
			else if constexpr (std::is_same<TP, float>::value)
				A[idxs[idx]] = sqrtf(A[idxs[idx]]);
			else if constexpr (std::is_same<TP, double>::value)
				A[idxs[idx]] = sqrt(A[idxs[idx]]);
			else
				// Handle other types here
				printf("Unsupported type in kernelSetMat, where F = 22(sqrt)");
		}
		else if constexpr (OP == np::NP_F_POW){
			if constexpr (std::is_same<TP, int>::value)
				A[idxs[idx]] = static_cast<int>(powf(A[idxs[idx]], B[idx]));
			else if constexpr (std::is_same<TP, float>::value)
				A[idxs[idx]] = powf(A[idxs[idx]], B[idx]);
			else if constexpr (std::is_same<TP, double>::value)
				A[idxs[idx]] = pow(A[idxs[idx]], static_cast<double>(B[idx]));
			else
				// Handle other types here
				printf("Unsupported type in kernelSetMat F=23(power)");
		}
		else
			printf("ERROR! INVALID OPERATOR/FUNCTION IN kernelSet.\n");
		idx += grid_size;		
	}
}

// sets A[r][c] to A[r][c] OP B, for r, c in zip(r_idxs, c_idxs)
template <typename TP, np::Operation OP>
__global__ void kernelSetMat(TP *A, const int rdin, const TP B, const int *r_idxs, const int *c_idxs, const int size){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int grid_size = blockDim.x * gridDim.x;

	while(idx < size){
		if constexpr (OP == np::NP_OP_ADD)
			A[r_idxs[idx] * rdin + c_idxs[idx]] += B;
		else if constexpr (OP == np::NP_OP_SUB)
			A[r_idxs[idx] * rdin + c_idxs[idx]] -= B;
		else if constexpr (OP == np::NP_OP_MUL)
			A[r_idxs[idx] * rdin + c_idxs[idx]] *= B;
		else if constexpr (OP == np::NP_OP_DIV)
			A[r_idxs[idx] * rdin + c_idxs[idx]] /= B;
		else if constexpr (OP == np::NP_OP_LESS_THAN)
			A[r_idxs[idx] * rdin + c_idxs[idx]] = A[r_idxs[idx] * rdin + c_idxs[idx]] < B;
		else if constexpr (OP == np::NP_OP_LESS_THAN_EQ)
			A[r_idxs[idx] * rdin + c_idxs[idx]] = A[r_idxs[idx] * rdin + c_idxs[idx]] <= B;
		else if constexpr (OP == np::NP_OP_GREATER_THAN)
			A[r_idxs[idx] * rdin + c_idxs[idx]] = A[r_idxs[idx] * rdin + c_idxs[idx]] > B;
		else if constexpr (OP == np::NP_OP_GREATER_THAN_EQ)
			A[r_idxs[idx] * rdin + c_idxs[idx]] = A[r_idxs[idx] * rdin + c_idxs[idx]] >= B;
		else if constexpr (OP == np::NP_OP_EQ_EQ)
			A[r_idxs[idx] * rdin + c_idxs[idx]] = A[r_idxs[idx] * rdin + c_idxs[idx]] == B;
		else if constexpr (OP == np::NP_OP_NOT_EQ)
			A[r_idxs[idx] * rdin + c_idxs[idx]] = A[r_idxs[idx] * rdin + c_idxs[idx]] != B;
		else if constexpr (OP == np::NP_OP_MINIMUM)
			A[r_idxs[idx] * rdin + c_idxs[idx]] = (A[r_idxs[idx] * rdin + c_idxs[idx]] < B) ? A[r_idxs[idx] * rdin + c_idxs[idx]] : B;
		else if constexpr (OP == np::NP_OP_MAXIMUM)
			A[r_idxs[idx] * rdin + c_idxs[idx]] = (A[r_idxs[idx] * rdin + c_idxs[idx]] > B) ? A[r_idxs[idx] * rdin + c_idxs[idx]] : B;
		else if constexpr (OP == np::NP_OP_EQ)
			A[r_idxs[idx] * rdin + c_idxs[idx]] = B;
		else if constexpr (OP == np::NP_F_EXP)
			A[r_idxs[idx] * rdin + c_idxs[idx]] = expf(A[r_idxs[idx] * rdin + c_idxs[idx]]);
		else if constexpr (OP == np::NP_F_LOG)
			A[r_idxs[idx] * rdin + c_idxs[idx]] = logf(A[r_idxs[idx] * rdin + c_idxs[idx]]);
		else if constexpr (OP == np::NP_F_SQUARE)
			A[r_idxs[idx] * rdin + c_idxs[idx]] *= A[r_idxs[idx] * rdin + c_idxs[idx]];
		else if constexpr (OP == np::NP_F_SQRT)
		{
			if constexpr (std::is_same<TP, int>::value)
				A[r_idxs[idx] * rdin + c_idxs[idx]] = static_cast<int>(sqrtf(A[r_idxs[idx] * rdin + c_idxs[idx]]));
			else if constexpr (std::is_same<TP, float>::value)
				A[r_idxs[idx] * rdin + c_idxs[idx]] = sqrtf(A[r_idxs[idx] * rdin + c_idxs[idx]]);
			else if constexpr (std::is_same<TP, double>::value)
				A[r_idxs[idx] * rdin + c_idxs[idx]] = sqrt(A[r_idxs[idx] * rdin + c_idxs[idx]]);
			else
				// Handle other types here
				printf("Unsupported type in kernelSetMat, where F = 22(sqrt)");
		}
		else if constexpr (OP == np::NP_F_POW){
			if constexpr (std::is_same<TP, int>::value)
				A[r_idxs[idx] * rdin + c_idxs[idx]] = static_cast<int>(powf(A[r_idxs[idx] * rdin + c_idxs[idx]], B));
			else if constexpr (std::is_same<TP, float>::value)
				A[r_idxs[idx] * rdin + c_idxs[idx]] = powf(A[r_idxs[idx] * rdin + c_idxs[idx]], B);
			else if constexpr (std::is_same<TP, double>::value)
				A[r_idxs[idx] * rdin + c_idxs[idx]] = pow(A[r_idxs[idx] * rdin + c_idxs[idx]], static_cast<double>(B));
			else
				// Handle other types here
				printf("Unsupported type in kernelSetMat F=23(power)");
		}
		else
			printf("ERROR! INVALID OPERATOR/FUNCTION IN kernelSet.\n");
		idx += grid_size;		
	}
}

// sets A[r][c] to A[r][c] OP B[r][c], for r, c in zip(r_idxs, c_idxs)
template <typename TP, np::Operation OP>
__global__ void kernelSetMat(TP *A, const int rdin, const TP *B, const int *r_idxs, const int *c_idxs, const int size){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int grid_size = blockDim.x * gridDim.x;

	while(idx < size){
		if constexpr (OP == np::NP_OP_ADD)
			A[r_idxs[idx] * rdin + c_idxs[idx]] += B[idx];
		else if constexpr (OP == np::NP_OP_SUB)
			A[r_idxs[idx] * rdin + c_idxs[idx]] -= B[idx];
		else if constexpr (OP == np::NP_OP_MUL)
			A[r_idxs[idx] * rdin + c_idxs[idx]] *= B[idx];
		else if constexpr (OP == np::NP_OP_DIV)
			A[r_idxs[idx] * rdin + c_idxs[idx]] /= B[idx];
		else if constexpr (OP == np::NP_OP_LESS_THAN)
			A[r_idxs[idx] * rdin + c_idxs[idx]] = A[r_idxs[idx] * rdin + c_idxs[idx]] < B[idx];
		else if constexpr (OP == np::NP_OP_LESS_THAN_EQ)
			A[r_idxs[idx] * rdin + c_idxs[idx]] = A[r_idxs[idx] * rdin + c_idxs[idx]] <= B[idx];
		else if constexpr (OP == np::NP_OP_GREATER_THAN)
			A[r_idxs[idx] * rdin + c_idxs[idx]] = A[r_idxs[idx] * rdin + c_idxs[idx]] > B[idx];
		else if constexpr (OP == np::NP_OP_GREATER_THAN_EQ)
			A[r_idxs[idx] * rdin + c_idxs[idx]] = A[r_idxs[idx] * rdin + c_idxs[idx]] >= B[idx];
		else if constexpr (OP == np::NP_OP_EQ_EQ)
			A[r_idxs[idx] * rdin + c_idxs[idx]] = A[r_idxs[idx] * rdin + c_idxs[idx]] == B[idx];
		else if constexpr (OP == np::NP_OP_NOT_EQ)
			A[r_idxs[idx] * rdin + c_idxs[idx]] = A[r_idxs[idx] * rdin + c_idxs[idx]] != B[idx];
		else if constexpr (OP == np::NP_OP_MINIMUM)
			A[r_idxs[idx] * rdin + c_idxs[idx]] = (A[r_idxs[idx] * rdin + c_idxs[idx]] < B[idx]) ? A[r_idxs[idx] * rdin + c_idxs[idx]] : B[idx];
		else if constexpr (OP == np::NP_OP_MAXIMUM)
			A[r_idxs[idx] * rdin + c_idxs[idx]] = (A[r_idxs[idx] * rdin + c_idxs[idx]] > B[idx]) ? A[r_idxs[idx] * rdin + c_idxs[idx]] : B[idx];
		else if constexpr (OP == np::NP_OP_EQ)
			A[r_idxs[idx] * rdin + c_idxs[idx]] = B[idx];
		else if constexpr (OP == np::NP_F_EXP)
			A[r_idxs[idx] * rdin + c_idxs[idx]] = expf(A[r_idxs[idx] * rdin + c_idxs[idx]]);
		else if constexpr (OP == np::NP_F_LOG)
			A[r_idxs[idx] * rdin + c_idxs[idx]] = logf(A[r_idxs[idx] * rdin + c_idxs[idx]]);
		else if constexpr (OP == np::NP_F_SQUARE)
			A[r_idxs[idx] * rdin + c_idxs[idx]] *= A[r_idxs[idx] * rdin + c_idxs[idx]];
		else if constexpr (OP == np::NP_F_SQRT)
		{
			if constexpr (std::is_same<TP, int>::value)
				A[r_idxs[idx] * rdin + c_idxs[idx]] = static_cast<int>(sqrtf(A[r_idxs[idx] * rdin + c_idxs[idx]]));
			else if constexpr (std::is_same<TP, float>::value)
				A[r_idxs[idx] * rdin + c_idxs[idx]] = sqrtf(A[r_idxs[idx] * rdin + c_idxs[idx]]);
			else if constexpr (std::is_same<TP, double>::value)
				A[r_idxs[idx] * rdin + c_idxs[idx]] = sqrt(A[r_idxs[idx] * rdin + c_idxs[idx]]);
			else
				// Handle other types here
				printf("Unsupported type in kernelSetMat, where F = 22(sqrt)");
		}
		else if constexpr (OP == np::NP_F_POW){
			if constexpr (std::is_same<TP, int>::value)
				A[r_idxs[idx] * rdin + c_idxs[idx]] = static_cast<int>(powf(A[r_idxs[idx] * rdin + c_idxs[idx]], B[idx]));
			else if constexpr (std::is_same<TP, float>::value)
				A[r_idxs[idx] * rdin + c_idxs[idx]] = powf(A[r_idxs[idx] * rdin + c_idxs[idx]], B[idx]);
			else if constexpr (std::is_same<TP, double>::value)
				A[r_idxs[idx] * rdin + c_idxs[idx]] = pow(A[r_idxs[idx] * rdin + c_idxs[idx]], static_cast<double>(B[idx]));
			else
				// Handle other types here
				printf("Unsupported type in kernelSetMat F=23(power)");
		}
		else
			printf("ERROR! INVALID OPERATOR/FUNCTION IN kernelSet.\n");
		idx += grid_size;		
	}
}
// operator functions

// perform operator on corrosponding elements of 2 matrices
//  C = A op B
template <typename TP, np::Operation OP>
__global__ void kernelMatOpMat(const TP *A, const TP *B, TP *C, const int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int grid_size = gridDim.x * blockDim.x;

	while (idx < size)
	{
		if constexpr (OP == np::NP_OP_ADD)
			C[idx] = A[idx] + B[idx];
		else if constexpr (OP == np::NP_OP_SUB)
			C[idx] = A[idx] - B[idx];
		else if constexpr (OP == np::NP_OP_MUL)
			C[idx] = A[idx] * B[idx];
		else if constexpr (OP == np::NP_OP_DIV)
			C[idx] = A[idx] / B[idx];
		else if constexpr (OP == np::NP_OP_LESS_THAN)
			C[idx] = A[idx] < B[idx];
		else if constexpr (OP == np::NP_OP_LESS_THAN_EQ)
			C[idx] = A[idx] <= B[idx];
		else if constexpr (OP == np::NP_OP_GREATER_THAN)
			C[idx] = A[idx] > B[idx];
		else if constexpr (OP == np::NP_OP_GREATER_THAN_EQ)
			C[idx] = A[idx] >= B[idx];
		else if constexpr (OP == np::NP_OP_EQ_EQ)
			C[idx] = A[idx] == B[idx];
		else if constexpr (OP == np::NP_OP_NOT_EQ)
			C[idx] = A[idx] != B[idx];
		else if constexpr (OP == np::NP_OP_MINIMUM)
			C[idx] = (A[idx] < B[idx]) ? A[idx] : B[idx];
		else if constexpr (OP == np::NP_OP_MAXIMUM)
			C[idx] = (A[idx] > B[idx]) ? A[idx] : B[idx];
		else
			printf("ERROR! INVALID OPERATOR IN kernelMatOPMat.\n");
		idx += grid_size;
	}
}

// operator on matrix and a scalar.
//  C = A op Scal (broadcasting)
// op scalar to all values of the matrix
template <typename TP, np::Operation OP>
__global__ void kernelMatOpScalar(const TP *A, const TP Scal, TP *C, const int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int grid_size = gridDim.x * blockDim.x;

	while (idx < size)
	{
		if constexpr (OP == np::NP_OP_ADD)
			C[idx] = A[idx] + Scal;
		else if constexpr (OP == np::NP_OP_SUB)
			C[idx] = A[idx] - Scal;
		else if constexpr (OP == np::NP_OP_MUL)
			C[idx] = A[idx] * Scal;
		else if constexpr (OP == np::NP_OP_DIV)
			C[idx] = A[idx] / Scal;
		else if constexpr (OP == np::NP_OP_LESS_THAN)
			C[idx] = A[idx] < Scal;
		else if constexpr (OP == np::NP_OP_LESS_THAN_EQ)
			C[idx] = A[idx] <= Scal;
		else if constexpr (OP == np::NP_OP_GREATER_THAN)
			C[idx] = A[idx] > Scal;
		else if constexpr (OP == np::NP_OP_GREATER_THAN_EQ)
			C[idx] = A[idx] >= Scal;
		else if constexpr (OP == np::NP_OP_EQ_EQ)
			C[idx] = A[idx] == Scal;
		else if constexpr (OP == np::NP_OP_NOT_EQ)
			C[idx] = A[idx] != Scal;
		else if constexpr (OP == np::NP_OP_MINIMUM)
			C[idx] = (A[idx] < Scal) ? A[idx] : Scal;
		else if constexpr (OP == np::NP_OP_MAXIMUM)
			C[idx] = (A[idx] > Scal) ? A[idx] : Scal;
		else
			printf("ERROR! INVALID OPERATOR IN kernelMatOPScalar.\n");
		idx += grid_size;
	}
}
template <typename TP, np::Operation OP>
__global__ void kernelMatOpScalar(const TP *A, const TP *Scal_a, TP *C, const int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int grid_size = gridDim.x * blockDim.x;
	const TP Scal = Scal_a[0];
	while (idx < size)
	{
		if constexpr (OP == np::NP_OP_ADD)
			C[idx] = A[idx] + Scal;
		else if constexpr (OP == np::NP_OP_SUB)
			C[idx] = A[idx] - Scal;
		else if constexpr (OP == np::NP_OP_MUL)
			C[idx] = A[idx] * Scal;
		else if constexpr (OP == np::NP_OP_DIV)
			C[idx] = A[idx] / Scal;
		else if constexpr (OP == np::NP_OP_LESS_THAN)
			C[idx] = A[idx] < Scal;
		else if constexpr (OP == np::NP_OP_LESS_THAN_EQ)
			C[idx] = A[idx] <= Scal;
		else if constexpr (OP == np::NP_OP_GREATER_THAN)
			C[idx] = A[idx] > Scal;
		else if constexpr (OP == np::NP_OP_GREATER_THAN_EQ)
			C[idx] = A[idx] >= Scal;
		else if constexpr (OP == np::NP_OP_EQ_EQ)
			C[idx] = A[idx] == Scal;
		else if constexpr (OP == np::NP_OP_NOT_EQ)
			C[idx] = A[idx] != Scal;
		else if constexpr (OP == np::NP_OP_MINIMUM)
			C[idx] = (A[idx] < Scal) ? A[idx] : Scal;
		else if constexpr (OP == np::NP_OP_MAXIMUM)
			C[idx] = (A[idx] > Scal) ? A[idx] : Scal;
		else
			printf("ERROR! INVALID OPERATOR IN kernelMatOPScalar.\n");
		idx += grid_size;
	}
}

template <typename TP, np::Operation OP>
__global__ void kernelScalarOpMat(const TP Scal, const TP *A, TP *C, const int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int grid_size = gridDim.x * blockDim.x;

	while (idx < size)
	{
		if constexpr (OP == np::NP_OP_ADD)
			C[idx] = Scal + A[idx];
		else if constexpr (OP == np::NP_OP_SUB)
			C[idx] = Scal - A[idx];
		else if constexpr (OP == np::NP_OP_MUL)
			C[idx] = Scal * A[idx];
		else if constexpr (OP == np::NP_OP_DIV)
			C[idx] = Scal / A[idx];
		else if constexpr (OP == np::NP_OP_LESS_THAN)
			C[idx] = Scal < A[idx];
		else if constexpr (OP == np::NP_OP_LESS_THAN_EQ)
			C[idx] = Scal <= A[idx];
		else if constexpr (OP == np::NP_OP_GREATER_THAN)
			C[idx] = Scal > A[idx];
		else if constexpr (OP == np::NP_OP_GREATER_THAN_EQ)
			C[idx] = Scal >= A[idx];
		else if constexpr (OP == np::NP_OP_EQ_EQ)
			C[idx] = Scal == A[idx];
		else if constexpr (OP == np::NP_OP_NOT_EQ)
			C[idx] = Scal != A[idx];
		else if constexpr (OP == np::NP_OP_MINIMUM)
			C[idx] = (A[idx] < Scal) ? A[idx] : Scal;
		else if constexpr (OP == np::NP_OP_MAXIMUM)
			C[idx] = (A[idx] > Scal) ? A[idx] : Scal;
		else
			printf("ERROR! INVALID OPERATOR IN kernelScalarOpMat.\n");
		idx += grid_size;
	}
}
template <typename TP, np::Operation OP>
__global__ void kernelScalarOpMat(const TP *Scal_a, const TP *A, TP *C, const int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int grid_size = gridDim.x * blockDim.x;

	const TP Scal = Scal_a[0];

	while (idx < size)
	{
		if constexpr (OP == np::NP_OP_ADD)
			C[idx] = Scal + A[idx];
		else if constexpr (OP == np::NP_OP_SUB)
			C[idx] = Scal - A[idx];
		else if constexpr (OP == np::NP_OP_MUL)
			C[idx] = Scal * A[idx];
		else if constexpr (OP == np::NP_OP_DIV)
			C[idx] = Scal / A[idx];
		else if constexpr (OP == np::NP_OP_LESS_THAN)
			C[idx] = Scal < A[idx];
		else if constexpr (OP == np::NP_OP_LESS_THAN_EQ)
			C[idx] = Scal <= A[idx];
		else if constexpr (OP == np::NP_OP_GREATER_THAN)
			C[idx] = Scal > A[idx];
		else if constexpr (OP == np::NP_OP_GREATER_THAN_EQ)
			C[idx] = Scal >= A[idx];
		else if constexpr (OP == np::NP_OP_EQ_EQ)
			C[idx] = Scal == A[idx];
		else if constexpr (OP == np::NP_OP_NOT_EQ)
			C[idx] = Scal != A[idx];
		else if constexpr (OP == np::NP_OP_MINIMUM)
			C[idx] = (A[idx] < Scal) ? A[idx] : Scal;
		else if constexpr (OP == np::NP_OP_MAXIMUM)
			C[idx] = (A[idx] > Scal) ? A[idx] : Scal;
		else
			printf("ERROR! INVALID OPERATOR IN kernelScalarOpMat.\n");
		idx += grid_size;
	}
}

// op on matrix and vector, mat.rows = vec.dim
// C = A op V (broadcasting)
//  shapeA = M x N matrix
template <typename TP, np::Operation OP>
__global__ void kernelMatOpVecAlongCols(const TP *A, const TP *V, TP *C, const int size, const int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int grid_size = gridDim.x * blockDim.x;
	int r;
	while (idx < size)
	{
		r = idx / N;
		// int c = idx % N;
		if constexpr (OP == np::NP_OP_ADD)
			C[idx] = A[idx] + V[r];
		else if constexpr (OP == np::NP_OP_SUB)
			C[idx] = A[idx] - V[r];
		else if constexpr (OP == np::NP_OP_MUL)
			C[idx] = A[idx] * V[r];
		else if constexpr (OP == np::NP_OP_DIV)
			C[idx] = A[idx] / V[r];
		else if constexpr (OP == np::NP_OP_LESS_THAN)
			C[idx] = A[idx] < V[r];
		else if constexpr (OP == np::NP_OP_LESS_THAN_EQ)
			C[idx] = A[idx] <= V[r];
		else if constexpr (OP == np::NP_OP_GREATER_THAN)
			C[idx] = A[idx] > V[r];
		else if constexpr (OP == np::NP_OP_GREATER_THAN_EQ)
			C[idx] = A[idx] >= V[r];
		else if constexpr (OP == np::NP_OP_EQ_EQ)
			C[idx] = A[idx] == V[r];
		else if constexpr (OP == np::NP_OP_NOT_EQ)
			C[idx] = A[idx] != V[r];
		else if constexpr (OP == np::NP_OP_MINIMUM)
			C[idx] = (A[idx] < V[r]) ? A[idx] : V[r];
		else if constexpr (OP == np::NP_OP_MAXIMUM)
			C[idx] = (A[idx] > V[r]) ? A[idx] : V[r];
		else
			printf("ERROR! INVALID OPERATOR IN kernelScalarOpMat.\n");
		idx += grid_size;
	}
}
template <typename TP, np::Operation OP>
__global__ void kernelVecOpMatAlongCols(const TP *V, const TP *A, TP *C, const int size, const int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int grid_size = blockDim.x * gridDim.x;
	int r;
	while (idx < size)
	{
		r = idx / N;
		// int c = idx % N;
		if constexpr (OP == np::NP_OP_ADD)
			C[idx] = V[r] + A[idx];
		else if constexpr (OP == np::NP_OP_SUB)
			C[idx] = V[r] - A[idx];
		else if constexpr (OP == np::NP_OP_MUL)
			C[idx] = V[r] * A[idx];
		else if constexpr (OP == np::NP_OP_DIV)
			C[idx] = V[r] / A[idx];
		else if constexpr (OP == np::NP_OP_LESS_THAN)
			C[idx] = V[r] < A[idx];
		else if constexpr (OP == np::NP_OP_LESS_THAN_EQ)
			C[idx] = V[r] <= A[idx];
		else if constexpr (OP == np::NP_OP_GREATER_THAN)
			C[idx] = V[r] > A[idx];
		else if constexpr (OP == np::NP_OP_GREATER_THAN_EQ)
			C[idx] = V[r] >= A[idx];
		else if constexpr (OP == np::NP_OP_EQ_EQ)
			C[idx] = V[r] == A[idx];
		else if constexpr (OP == np::NP_OP_NOT_EQ)
			C[idx] = V[r] != A[idx];
		else if constexpr (OP == np::NP_OP_MINIMUM)
			C[idx] = (A[idx] < V[r]) ? A[idx] : V[r];
		else if constexpr (OP == np::NP_OP_MAXIMUM)
			C[idx] = (A[idx] > V[r]) ? A[idx] : V[r];
		else
			printf("ERROR! INVALID OPERATOR IN kernelScalarOpMat.\n");
		idx += grid_size;
	}
}

// operator on matrix and vector, mat.cols = vec.dim
// C = A op V (broadcasting)
// shapeA = M x N matrix
template <typename TP, np::Operation OP>
__global__ void kernelMatOpVecAlongRows(const TP *A, const TP *V, TP *C, const int size, const int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int grid_size = gridDim.x * blockDim.x;
	int c;
	while (idx < size)
	{
		// int r = idx / N;
		c = idx % N;
		if constexpr (OP == np::NP_OP_ADD)
			C[idx] = A[idx] + V[c];
		else if constexpr (OP == np::NP_OP_SUB)
			C[idx] = A[idx] - V[c];
		else if constexpr (OP == np::NP_OP_MUL)
			C[idx] = A[idx] * V[c];
		else if constexpr (OP == np::NP_OP_DIV)
			C[idx] = A[idx] / V[c];
		else if constexpr (OP == np::NP_OP_LESS_THAN)
			C[idx] = A[idx] < V[c];
		else if constexpr (OP == np::NP_OP_LESS_THAN_EQ)
			C[idx] = A[idx] <= V[c];
		else if constexpr (OP == np::NP_OP_GREATER_THAN)
			C[idx] = A[idx] > V[c];
		else if constexpr (OP == np::NP_OP_GREATER_THAN_EQ)
			C[idx] = A[idx] >= V[c];
		else if constexpr (OP == np::NP_OP_EQ_EQ)
			C[idx] = A[idx] == V[c];
		else if constexpr (OP == np::NP_OP_NOT_EQ)
			C[idx] = A[idx] != V[c];
		else if constexpr (OP == np::NP_OP_MINIMUM)
			C[idx] = (A[idx] < V[c]) ? A[idx] : V[c];
		else if constexpr (OP == np::NP_OP_MAXIMUM)
			C[idx] = (A[idx] > V[c]) ? A[idx] : V[c];
		else
			printf("ERROR! INVALID OPERATOR IN kernelScalarOpMat.\n");
		idx += grid_size;
	}
}

template <typename TP, np::Operation OP>
__global__ void kernelVecOpMatAlongRows(const TP *V, const TP *A, TP *C, const int size, const int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int grid_size = blockDim.x * gridDim.x;
	int c;
	while (idx < size)
	{
		// int r = idx / N;
		c = idx % N;
		if constexpr (OP == np::NP_OP_ADD)
			C[idx] = V[c] + A[idx];
		else if constexpr (OP == np::NP_OP_SUB)
			C[idx] = V[c] - A[idx];
		else if constexpr (OP == np::NP_OP_MUL)
			C[idx] = V[c] * A[idx];
		else if constexpr (OP == np::NP_OP_DIV)
			C[idx] = V[c] / A[idx];
		else if constexpr (OP == np::NP_OP_LESS_THAN)
			C[idx] = V[c] < A[idx];
		else if constexpr (OP == np::NP_OP_LESS_THAN_EQ)
			C[idx] = V[c] <= A[idx];
		else if constexpr (OP == np::NP_OP_GREATER_THAN)
			C[idx] = V[c] > A[idx];
		else if constexpr (OP == np::NP_OP_GREATER_THAN_EQ)
			C[idx] = V[c] >= A[idx];
		else if constexpr (OP == np::NP_OP_EQ_EQ)
			C[idx] = V[c] == A[idx];
		else if constexpr (OP == np::NP_OP_NOT_EQ)
			C[idx] = V[c] != A[idx];
		else if constexpr (OP == np::NP_OP_MINIMUM)
			C[idx] = (A[idx] < V[c]) ? A[idx] : V[c];
		else if constexpr (OP == np::NP_OP_MAXIMUM)
			C[idx] = (A[idx] > V[c]) ? A[idx] : V[c];
		else
			printf("ERROR! INVALID OPERATOR IN kernelScalarOpMat.\n");
		idx += grid_size;
	}
}

// npfunctions
// A = MxN
// C = MxN
// Ci = F(Ai)
template <typename TP, np::Operation OP>
__global__ void kernelFMat(const TP *A, TP *C, const int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int grid_size = blockDim.x * gridDim.x;

	while (idx < size)
	{
		if constexpr (OP == np::NP_F_EXP)
			C[idx] = expf(A[idx]);
		else if constexpr (OP == np::NP_F_LOG)
			C[idx] = logf(A[idx]);
		else if constexpr (OP == np::NP_F_SQUARE)
			C[idx] = A[idx] * A[idx];
		else if constexpr (OP == np::NP_F_SQRT)
		{
			if constexpr (std::is_same<TP, int>::value)
				C[idx] = static_cast<int>(sqrtf(A[idx]));
			else if constexpr (std::is_same<TP, float>::value)
				C[idx] = sqrtf(A[idx]);
			else if constexpr (std::is_same<TP, double>::value)
				C[idx] = sqrt(A[idx]);
			else
				// Handle other types here
				printf("Unsupported type in kernelFMat, where F = 22(sqrt)");
		}
		else
			printf("Unsupported function type in kernelFMat\n");
		idx += grid_size;
	}
}

// np.pow
// A = MxN
// C = MxN
// Ci = square(Ai)
template <typename TP>
__global__ void kernelPowMat(const TP *A, const float power, TP *C, const int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int grid_size = blockDim.x * gridDim.x;
	while (idx < size)
	{
		if constexpr (std::is_same<TP, int>::value)
		{
			C[idx] = static_cast<int>(powf(A[idx], power));
		}
		else if constexpr (std::is_same<TP, float>::value)
		{
			C[idx] = powf(A[idx], power);
		}
		else if constexpr (std::is_same<TP, double>::value)
		{
			C[idx] = pow(A[idx], static_cast<double>(power));
		}
		else
		{
			// Handle other types here
			printf("Unsupported type in kernelPowMat");
		}
		idx += grid_size;
	}
}

// REDUCTION
template <typename TP, np::Operation OP>
__device__ void kernelWarpReduceF(volatile TP *s_A, const int tid)
{ // warp reduce for kernel
	if constexpr (OP == np::NP_REDUCE_SUM)
	{
		s_A[tid] += s_A[tid + 32];
		s_A[tid] += s_A[tid + 16];
		s_A[tid] += s_A[tid + 8];
		s_A[tid] += s_A[tid + 4];
		s_A[tid] += s_A[tid + 2];
		s_A[tid] += s_A[tid + 1];
	}
	else if constexpr (OP == np::NP_REDUCE_MIN)
	{
		s_A[tid] = min(s_A[tid], s_A[tid + 32]);
		s_A[tid] = min(s_A[tid], s_A[tid + 16]);
		s_A[tid] = min(s_A[tid], s_A[tid + 8]);
		s_A[tid] = min(s_A[tid], s_A[tid + 4]);
		s_A[tid] = min(s_A[tid], s_A[tid + 2]);
		s_A[tid] = min(s_A[tid], s_A[tid + 1]);
	}
	else if constexpr (OP == np::NP_REDUCE_MAX)
	{
		s_A[tid] = max(s_A[tid], s_A[tid + 32]);
		s_A[tid] = max(s_A[tid], s_A[tid + 16]);
		s_A[tid] = max(s_A[tid], s_A[tid + 8]);
		s_A[tid] = max(s_A[tid], s_A[tid + 4]);
		s_A[tid] = max(s_A[tid], s_A[tid + 2]);
		s_A[tid] = max(s_A[tid], s_A[tid + 1]);
	}
	else
		printf("INVALID ARGUMENT! in kernelWarpReduceF\n");
}

// warp unroll
template <typename TP, int BLOCK_SIZE, np::Operation OP>
__global__ void kernelReduceF(const TP *A, TP *output, const int size)
{
	if constexpr (OP != np::NP_REDUCE_SUM && OP != np::NP_REDUCE_MIN && OP != np::NP_REDUCE_MAX)
	{
		printf("INVALID ARGUMENT! in kernelReduceF\n");
		return;
	}
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	int idx = bx * BLOCK_SIZE * 2 + tx;
	const int grid_size = BLOCK_SIZE * 2 * gridDim.x;
	__shared__ TP s_A[BLOCK_SIZE];

	if constexpr (OP == np::NP_REDUCE_SUM)
		s_A[tx] = 0;
	else if constexpr (OP == np::NP_REDUCE_MIN)
		s_A[tx] = INT_MAX;
	else if constexpr (OP == np::NP_REDUCE_MAX)
		s_A[tx] = INT_MIN;
	// assume 1 hi grid launch kr rha h tu
	while (idx < size)
	{
		if constexpr (OP == np::NP_REDUCE_SUM)
			s_A[tx] += (A[idx] + ((idx + BLOCK_SIZE < size) ? A[idx + BLOCK_SIZE] : 0));
		else if constexpr (OP == np::NP_REDUCE_MIN)
		{
			s_A[tx] = min(s_A[tx], A[idx]);
			if (idx + BLOCK_SIZE < size)
				s_A[tx] = min(s_A[tx], A[idx + BLOCK_SIZE]);
		}
		else if constexpr (OP == np::NP_REDUCE_MAX)
		{
			s_A[tx] = max(s_A[tx], A[idx]);
			if (idx + BLOCK_SIZE < size)
				s_A[tx] = max(s_A[tx], A[idx + BLOCK_SIZE]);
		}

		idx += grid_size;
	}
	__syncthreads();

	if constexpr (BLOCK_SIZE > 511)
	{
		if (tx < 256)
		{
			if constexpr (OP == np::NP_REDUCE_SUM)
				s_A[tx] += s_A[tx + 256];
			else if constexpr (OP == np::NP_REDUCE_MIN)
				s_A[tx] = min(s_A[tx], s_A[tx + 256]);
			else if constexpr (OP == np::NP_REDUCE_MAX)
				s_A[tx] = max(s_A[tx], s_A[tx + 256]);
		}
		__syncthreads();
	}

	if constexpr (BLOCK_SIZE > 255)
	{
		if (tx < 128)
		{
			if constexpr (OP == np::NP_REDUCE_SUM)
				s_A[tx] += s_A[tx + 128];
			else if constexpr (OP == np::NP_REDUCE_MIN)
				s_A[tx] = min(s_A[tx], s_A[tx + 128]);
			else if constexpr (OP == np::NP_REDUCE_MAX)
				s_A[tx] = max(s_A[tx], s_A[tx + 128]);
		}
		__syncthreads();
	}
	if constexpr (BLOCK_SIZE > 127)
	{
		if (tx < 64)
		{
			if constexpr (OP == np::NP_REDUCE_SUM)
				s_A[tx] += s_A[tx + 64];
			else if constexpr (OP == np::NP_REDUCE_MIN)
				s_A[tx] = min(s_A[tx], s_A[tx + 64]);
			else if constexpr (OP == np::NP_REDUCE_MAX)
				s_A[tx] = max(s_A[tx], s_A[tx + 64]);
		}
		__syncthreads();
	}

	if (tx < 32)
		kernelWarpReduceF<TP, OP>(s_A, tx);

	if (tx == 0)
		output[bx] = s_A[0];
}

// warp unroll
template <typename TP, int BLOCK_SIZE, np::Operation OP>
__global__ void kernelReduceFAxis1(const TP *A, TP *output, const int size, const int rows)
{
	if constexpr (OP != np::NP_REDUCE_SUM && OP != np::NP_REDUCE_MIN && OP != np::NP_REDUCE_MAX)
	{
		printf("INVALID ARGUMENT! in kernelReduceF\n");
		return;
	}
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	int _idx = bx * BLOCK_SIZE * 2 + tx;
	const int grid_size = BLOCK_SIZE * 2 * gridDim.x;
	__shared__ TP s_A[BLOCK_SIZE];

	for(int r = 0; r < rows; ++r){
		int idx = _idx;
		if constexpr (OP == np::NP_REDUCE_SUM)
			s_A[tx] = 0;
		else if constexpr (OP == np::NP_REDUCE_MIN)
			s_A[tx] = INT_MAX;
		else if constexpr (OP == np::NP_REDUCE_MAX)
			s_A[tx] = INT_MIN;
		// assume 1 hi grid launch kr rha h tu
		while (idx < size)
		{
			if constexpr (OP == np::NP_REDUCE_SUM)
				s_A[tx] += (A[idx] + ((idx + BLOCK_SIZE < size) ? A[idx + BLOCK_SIZE] : 0));
			else if constexpr (OP == np::NP_REDUCE_MIN)
			{
				s_A[tx] = min(s_A[tx], A[idx]);
				if (idx + BLOCK_SIZE < size)
					s_A[tx] = min(s_A[tx], A[idx + BLOCK_SIZE]);
			}
			else if constexpr (OP == np::NP_REDUCE_MAX)
			{
				s_A[tx] = max(s_A[tx], A[idx]);
				if (idx + BLOCK_SIZE < size)
					s_A[tx] = max(s_A[tx], A[idx + BLOCK_SIZE]);
			}

			idx += grid_size;
		}
		__syncthreads();

		if constexpr (BLOCK_SIZE > 511)
		{
			if (tx < 256)
			{
				if constexpr (OP == np::NP_REDUCE_SUM)
					s_A[tx] += s_A[tx + 256];
				else if constexpr (OP == np::NP_REDUCE_MIN)
					s_A[tx] = min(s_A[tx], s_A[tx + 256]);
				else if constexpr (OP == np::NP_REDUCE_MAX)
					s_A[tx] = max(s_A[tx], s_A[tx + 256]);
			}
			__syncthreads();
		}

		if constexpr (BLOCK_SIZE > 255)
		{
			if (tx < 128)
			{
				if constexpr (OP == np::NP_REDUCE_SUM)
					s_A[tx] += s_A[tx + 128];
				else if constexpr (OP == np::NP_REDUCE_MIN)
					s_A[tx] = min(s_A[tx], s_A[tx + 128]);
				else if constexpr (OP == np::NP_REDUCE_MAX)
					s_A[tx] = max(s_A[tx], s_A[tx + 128]);
			}
			__syncthreads();
		}
		if constexpr (BLOCK_SIZE > 127)
		{
			if (tx < 64)
			{
				if constexpr (OP == np::NP_REDUCE_SUM)
					s_A[tx] += s_A[tx + 64];
				else if constexpr (OP == np::NP_REDUCE_MIN)
					s_A[tx] = min(s_A[tx], s_A[tx + 64]);
				else if constexpr (OP == np::NP_REDUCE_MAX)
					s_A[tx] = max(s_A[tx], s_A[tx + 64]);
			}
			__syncthreads();
		}

		if (tx < 32)
			kernelWarpReduceF<TP, OP>(s_A, tx);

		if (tx == 0)
			output[bx] = s_A[0];
		A+= size;
		output += gridDim.x;
		__syncthreads();
	}
}


template <typename TP, np::Operation OP>
__device__ void kernelWarpReduceArgF(volatile TP *s_A, volatile int *s_Idx, const int tid)
{ // warp reduce for kernel
	if constexpr (OP == np::NP_REDUCE_ARGMIN)
	{
		if (s_A[tid] > s_A[tid + 32])
		{
			s_A[tid] = s_A[tid + 32];
			s_Idx[tid] = s_Idx[tid + 32];
		}
		if (s_A[tid] > s_A[tid + 16])
		{
			s_A[tid] = s_A[tid + 16];
			s_Idx[tid] = s_Idx[tid + 16];
		}
		if (s_A[tid] > s_A[tid + 16])
		{
			s_A[tid] = s_A[tid + 16];
			s_Idx[tid] = s_Idx[tid + 16];
		}
		if (s_A[tid] > s_A[tid + 8])
		{
			s_A[tid] = s_A[tid + 8];
			s_Idx[tid] = s_Idx[tid + 8];
		}
		if (s_A[tid] > s_A[tid + 4])
		{
			s_A[tid] = s_A[tid + 4];
			s_Idx[tid] = s_Idx[tid + 4];
		}
		if (s_A[tid] > s_A[tid + 2])
		{
			s_A[tid] = s_A[tid + 2];
			s_Idx[tid] = s_Idx[tid + 2];
		}
		if (s_A[tid] > s_A[tid + 1])
		{
			s_A[tid] = s_A[tid + 1];
			s_Idx[tid] = s_Idx[tid + 1];
		}
	}
	else if constexpr (OP == np::NP_REDUCE_ARGMAX)
	{
		if (s_A[tid] < s_A[tid + 32])
		{
			s_A[tid] = s_A[tid + 32];
			s_Idx[tid] = s_Idx[tid + 32];
		}
		if (s_A[tid] < s_A[tid + 16])
		{
			s_A[tid] = s_A[tid + 16];
			s_Idx[tid] = s_Idx[tid + 16];
		}
		if (s_A[tid] < s_A[tid + 16])
		{
			s_A[tid] = s_A[tid + 16];
			s_Idx[tid] = s_Idx[tid + 16];
		}
		if (s_A[tid] < s_A[tid + 8])
		{
			s_A[tid] = s_A[tid + 8];
			s_Idx[tid] = s_Idx[tid + 8];
		}
		if (s_A[tid] < s_A[tid + 4])
		{
			s_A[tid] = s_A[tid + 4];
			s_Idx[tid] = s_Idx[tid + 4];
		}
		if (s_A[tid] < s_A[tid + 2])
		{
			s_A[tid] = s_A[tid + 2];
			s_Idx[tid] = s_Idx[tid + 2];
		}
		if (s_A[tid] < s_A[tid + 1])
		{
			s_A[tid] = s_A[tid + 1];
			s_Idx[tid] = s_Idx[tid + 1];
		}
	}
}

// warp unroll
template <typename TP, int BLOCK_SIZE, np::Operation OP>
__global__ void kernelReduceArgF(const TP *A, TP *outputMax, int *outputIdx, const int size)
{
	if constexpr (OP != np::NP_REDUCE_ARGMIN && OP != np::NP_REDUCE_ARGMAX)
	{
		printf("INVALID ARGUMENT! in kernelReduceArgF\n");
		return;
	}
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	int idx = bx * BLOCK_SIZE * 2 + tx;
	const int grid_size = BLOCK_SIZE * 2 * gridDim.x;
	__shared__ TP s_A[BLOCK_SIZE];
	__shared__ int s_Idx[BLOCK_SIZE];

	if constexpr (OP == np::NP_REDUCE_ARGMIN)
		s_A[tx] = INT_MAX;
	else if constexpr (OP == np::NP_REDUCE_ARGMAX)
		s_A[tx] = INT_MIN;
	s_Idx[tx] = -1;
	// assume 1 hi grid launch kr rha h tu
	while (idx < size)
	{
		if constexpr (OP == np::NP_REDUCE_ARGMIN)
		{
			if (s_A[tx] > A[idx])
			{
				s_A[tx] = A[idx];
				s_Idx[tx] = idx;
			}
			if (idx + BLOCK_SIZE < size)
			{
				if (s_A[tx] > A[idx + BLOCK_SIZE])
				{
					s_A[tx] = A[idx + BLOCK_SIZE];
					s_Idx[tx] = idx + BLOCK_SIZE;
				}
			}
		}
		else if constexpr (OP == np::NP_REDUCE_ARGMAX)
		{
			if (s_A[tx] < A[idx])
			{
				s_A[tx] = A[idx];
				s_Idx[tx] = idx;
			}
			if (idx + BLOCK_SIZE < size)
			{
				if (s_A[tx] < A[idx + BLOCK_SIZE])
				{
					s_A[tx] = A[idx + BLOCK_SIZE];
					s_Idx[tx] = idx + BLOCK_SIZE;
				}
			}
		}

		idx += grid_size;
	}
	__syncthreads();
	
	if constexpr (BLOCK_SIZE > 511)
	{
		if (tx < 256)
		{
			if constexpr (OP == np::NP_REDUCE_ARGMIN)
			{
				if (s_A[tx] > s_A[tx + 256])
				{
					s_A[tx] = s_A[tx + 256];
					s_Idx[tx] = s_Idx[tx + 256];
				}
			}
			else if constexpr (OP == np::NP_REDUCE_ARGMAX)
			{
				if (s_A[tx] < s_A[tx + 256])
				{
					s_A[tx] = s_A[tx + 256];
					s_Idx[tx] = s_Idx[tx + 256];

				}
			}
		}
		__syncthreads();
	}

	if constexpr (BLOCK_SIZE > 255)
	{
		if (tx < 128)
		{
			if constexpr (OP == np::NP_REDUCE_ARGMIN)
			{
				if (s_A[tx] > s_A[tx + 128])
				{
					s_A[tx] = s_A[tx + 128];
					s_Idx[tx] = s_Idx[tx + 128];
				}
			}
			else if constexpr (OP == np::NP_REDUCE_ARGMAX)
			{
				if (s_A[tx] < s_A[tx + 128])
				{
					s_A[tx] = s_A[tx + 128];
					s_Idx[tx] = s_Idx[tx + 128];
				}
			}
		}
		__syncthreads();
	}

	if constexpr (BLOCK_SIZE > 127)
	{
		if (tx < 64)
		{
			if constexpr (OP == np::NP_REDUCE_ARGMIN)
			{
				if (s_A[tx] > s_A[tx + 64])
				{
					s_A[tx] = s_A[idx + 64];
					s_Idx[tx] = s_Idx[tx + 64];
				}
			}
			else if constexpr (OP == np::NP_REDUCE_ARGMAX)
			{
				if (s_A[tx] < s_A[tx + 64])
				{
					s_A[tx] = s_A[tx + 64];
					s_Idx[tx] = s_Idx[tx + 64];
				}
			}
		}
		__syncthreads();
	}

	if (tx < 32)
		kernelWarpReduceArgF<TP, OP>(s_A, s_Idx, tx);

	if (tx == 0)
	{
		outputMax[bx] = s_A[0];
		outputIdx[bx] = s_Idx[0];
	}
}

template <typename TP, int BLOCK_SIZE, np::Operation OP>
__global__ void kernelReduceArgFAxis1(const TP *A, TP *outputMax, int *outputIdx, const int size, const int rows){
	if constexpr (OP != np::NP_REDUCE_ARGMIN && OP != np::NP_REDUCE_ARGMAX)
	{
		printf("INVALID ARGUMENT! in kernelReduceArgF\n");
		return;
	}
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	int _idx = bx * BLOCK_SIZE * 2 + tx;
	const int grid_size = BLOCK_SIZE * 2 * gridDim.x;
	__shared__ TP s_A[BLOCK_SIZE];
	__shared__ int s_Idx[BLOCK_SIZE];

	for(int r = 0; r < rows; ++r){
		int idx = _idx;
		if constexpr (OP == np::NP_REDUCE_ARGMIN)
			s_A[tx] = INT_MAX;
		else if constexpr (OP == np::NP_REDUCE_ARGMAX)
			s_A[tx] = INT_MIN;
		s_Idx[tx] = -1;
		// assume 1 hi grid launch kr rha h tu
		while (idx < size)
		{
			if constexpr (OP == np::NP_REDUCE_ARGMIN)
			{
				if (s_A[tx] > A[idx])
				{
					s_A[tx] = A[idx];
					s_Idx[tx] = idx;
				}
				if (idx + BLOCK_SIZE < size)
				{
					if (s_A[tx] > A[idx + BLOCK_SIZE])
					{
						s_A[tx] = A[idx + BLOCK_SIZE];
						s_Idx[tx] = idx + BLOCK_SIZE;
					}
				}
			}
			else if constexpr (OP == np::NP_REDUCE_ARGMAX)
			{
				if (s_A[tx] < A[idx])
				{
					s_A[tx] = A[idx];
					s_Idx[tx] = idx;
				}
				if (idx + BLOCK_SIZE < size)
				{
					if (s_A[tx] < A[idx + BLOCK_SIZE])
					{
						s_A[tx] = A[idx + BLOCK_SIZE];
						s_Idx[tx] = idx + BLOCK_SIZE;
					}
				}
			}

			idx += grid_size;
		}
		__syncthreads();
		
		if constexpr (BLOCK_SIZE > 511)
		{
			if (tx < 256)
			{
				if constexpr (OP == np::NP_REDUCE_ARGMIN)
				{
					if (s_A[tx] > s_A[tx + 256])
					{
						s_A[tx] = s_A[tx + 256];
						s_Idx[tx] = s_Idx[tx + 256];
					}
				}
				else if constexpr (OP == np::NP_REDUCE_ARGMAX)
				{
					if (s_A[tx] < s_A[tx + 256])
					{
						s_A[tx] = s_A[tx + 256];
						s_Idx[tx] = s_Idx[tx + 256];

					}
				}
			}
			__syncthreads();
		}

		if constexpr (BLOCK_SIZE > 255)
		{
			if (tx < 128)
			{
				if constexpr (OP == np::NP_REDUCE_ARGMIN)
				{
					if (s_A[tx] > s_A[tx + 128])
					{
						s_A[tx] = s_A[tx + 128];
						s_Idx[tx] = s_Idx[tx + 128];
					}
				}
				else if constexpr (OP == np::NP_REDUCE_ARGMAX)
				{
					if (s_A[tx] < s_A[tx + 128])
					{
						s_A[tx] = s_A[tx + 128];
						s_Idx[tx] = s_Idx[tx + 128];
					}
				}
			}
			__syncthreads();
		}

		if constexpr (BLOCK_SIZE > 127)
		{
			if (tx < 64)
			{
				if constexpr (OP == np::NP_REDUCE_ARGMIN)
				{
					if (s_A[tx] > s_A[tx + 64])
					{
						s_A[tx] = s_A[idx + 64];
						s_Idx[tx] = s_Idx[tx + 64];
					}
				}
				else if constexpr (OP == np::NP_REDUCE_ARGMAX)
				{
					if (s_A[tx] < s_A[tx + 64])
					{
						s_A[tx] = s_A[tx + 64];
						s_Idx[tx] = s_Idx[tx + 64];
					}
				}
			}
			__syncthreads();
		}

		if (tx < 32)
			kernelWarpReduceArgF<TP, OP>(s_A, s_Idx, tx);

		if (tx == 0)
		{
			outputMax[bx] = s_A[0];
			outputIdx[bx] = s_Idx[0];
		}
		A += size;
		outputIdx += gridDim.x;
		outputMax += gridDim.x;
	}
}

// second reduction k time p -> idx serial nhi h, to ek idx ka bhi array dena hoga
template <typename TP, int BLOCK_SIZE, np::Operation OP>
__global__ void kernelReduceArgF(const TP *A, const int *A_idx, TP *outputMax, int *outputIdx, const int size)
{
	if constexpr (OP != np::NP_REDUCE_ARGMIN && OP != np::NP_REDUCE_ARGMAX)
	{
		printf("INVALID ARGUMENT! in kernelReduceArgF\n");
		return;
	}
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	int idx = bx * BLOCK_SIZE * 2 + tx;
	const int grid_size = BLOCK_SIZE * 2 * gridDim.x;
	__shared__ TP s_A[BLOCK_SIZE];
	__shared__ int s_Idx[BLOCK_SIZE];

	if constexpr (OP == np::NP_REDUCE_ARGMIN)
		s_A[tx] = INT_MAX;
	else if constexpr (OP == np::NP_REDUCE_ARGMAX)
		s_A[tx] = INT_MIN;
	s_Idx[tx] = -1;

	// assume 1 hi grid launch kr rha h tu
	while (idx < size)
	{
		if constexpr (OP == np::NP_REDUCE_ARGMIN)
		{
			if (s_A[tx] > A[idx])
			{
				s_A[tx] = A[idx];
				s_Idx[tx] = A_idx[idx];
			}
			if (idx + BLOCK_SIZE < size && s_A[tx] > A[idx + BLOCK_SIZE])
			{
				s_A[tx] = A[idx + BLOCK_SIZE];
				s_Idx[tx] = A_idx[idx + BLOCK_SIZE];
			}
		}
		else if constexpr (OP == np::NP_REDUCE_ARGMAX)
		{
			if (s_A[tx] < A[idx])
			{
				s_A[tx] = A[idx];
				s_Idx[tx] = A_idx[idx];
			}
			if (idx + BLOCK_SIZE < size)
			{
				if (s_A[tx] < A[idx + BLOCK_SIZE])
				{
					s_A[tx] = A[idx + BLOCK_SIZE];
					s_Idx[tx] = A_idx[idx + BLOCK_SIZE];
				}
			}
		}

		idx += grid_size;
	}
	__syncthreads();

	if constexpr (BLOCK_SIZE > 511)
	{
		if (tx < 256)
		{
			if constexpr (OP == np::NP_REDUCE_ARGMIN)
			{
				if (s_A[tx] > s_A[tx + 256])
				{
					s_A[tx] = s_A[tx + 256];
					s_Idx[tx] = s_Idx[tx + 256];
				}
			}
			else if constexpr (OP == np::NP_REDUCE_ARGMAX)
			{
				if (s_A[tx] < s_A[tx + 256])
				{
					s_A[tx] = s_A[tx + 256];
					s_Idx[tx] = s_Idx[tx + 256];
				}
			}
		}
		__syncthreads();
	}

	if constexpr (BLOCK_SIZE > 255)
	{
		if (tx < 128)
		{
			if constexpr (OP == np::NP_REDUCE_ARGMIN)
			{
				if (s_A[tx] > s_A[tx + 128])
				{
					s_A[tx] = s_A[tx + 128];
					s_Idx[tx] = s_Idx[tx + 128];
				}
			}
			else if constexpr (OP == np::NP_REDUCE_ARGMAX)
			{
				if (s_A[tx] < s_A[tx + 128])
				{
					s_A[tx] = s_A[tx + 128];
					s_Idx[tx] = s_Idx[tx + 128];
				}
			}
		}
		__syncthreads();
	}
	if constexpr (BLOCK_SIZE > 127)
	{
		if (tx < 64)
		{
			if constexpr (OP == np::NP_REDUCE_ARGMIN)
			{
				if (s_A[tx] > s_A[tx + 64])
				{
					s_A[tx] = s_A[tx + 64];
					s_Idx[tx] = s_A[tx + 64];
				}
			}
			else if constexpr (OP == np::NP_REDUCE_ARGMAX)
			{
				if (s_A[tx] < s_A[tx + 64])
				{
					s_A[tx] = s_A[tx + 64];
					s_Idx[tx] = s_Idx[tx + 64];
				}
			}
		}
		__syncthreads();
	}

	if (tx < 32)
		kernelWarpReduceArgF<TP, OP>(s_A, s_Idx, tx);

	if (tx == 0)
	{
		outputMax[bx] = s_A[0];
		outputIdx[bx] = s_Idx[0];
	}
}


// second reduction k time p -> idx serial nhi h, to ek idx ka bhi array dena hoga
template <typename TP, int BLOCK_SIZE, np::Operation OP>
__global__ void kernelReduceArgFAxis1(const TP *A, const int *A_idx, TP *outputMax, int *outputIdx, const int size, const int rows)
{
	if constexpr (OP != np::NP_REDUCE_ARGMIN && OP != np::NP_REDUCE_ARGMAX)
	{
		printf("INVALID ARGUMENT! in kernelReduceArgF\n");
		return;
	}
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	int _idx = bx * BLOCK_SIZE * 2 + tx;
	const int grid_size = BLOCK_SIZE * 2 * gridDim.x;
	__shared__ TP s_A[BLOCK_SIZE];
	__shared__ int s_Idx[BLOCK_SIZE];


	for(int r = 0; r < rows; ++r){
		int idx = _idx;
		if constexpr (OP == np::NP_REDUCE_ARGMIN)
			s_A[tx] = INT_MAX;
		else if constexpr (OP == np::NP_REDUCE_ARGMAX)
			s_A[tx] = INT_MIN;
		s_Idx[tx] = -1;

		// assume 1 hi grid launch kr rha h tu
		while (idx < size)
		{
			if constexpr (OP == np::NP_REDUCE_ARGMIN)
			{
				if (s_A[tx] > A[idx])
				{
					s_A[tx] = A[idx];
					s_Idx[tx] = A_idx[idx];
				}
				if (idx + BLOCK_SIZE < size && s_A[tx] > A[idx + BLOCK_SIZE])
				{
					s_A[tx] = A[idx + BLOCK_SIZE];
					s_Idx[tx] = A_idx[idx + BLOCK_SIZE];
				}
			}
			else if constexpr (OP == np::NP_REDUCE_ARGMAX)
			{
				if (s_A[tx] < A[idx])
				{
					s_A[tx] = A[idx];
					s_Idx[tx] = A_idx[idx];
				}
				if (idx + BLOCK_SIZE < size)
				{
					if (s_A[tx] < A[idx + BLOCK_SIZE])
					{
						s_A[tx] = A[idx + BLOCK_SIZE];
						s_Idx[tx] = A_idx[idx + BLOCK_SIZE];
					}
				}
			}

			idx += grid_size;
		}
		__syncthreads();

		if constexpr (BLOCK_SIZE > 511)
		{
			if (tx < 256)
			{
				if constexpr (OP == np::NP_REDUCE_ARGMIN)
				{
					if (s_A[tx] > s_A[tx + 256])
					{
						s_A[tx] = s_A[tx + 256];
						s_Idx[tx] = s_Idx[tx + 256];
					}
				}
				else if constexpr (OP == np::NP_REDUCE_ARGMAX)
				{
					if (s_A[tx] < s_A[tx + 256])
					{
						s_A[tx] = s_A[tx + 256];
						s_Idx[tx] = s_Idx[tx + 256];
					}
				}
			}
			__syncthreads();
		}

		if constexpr (BLOCK_SIZE > 255)
		{
			if (tx < 128)
			{
				if constexpr (OP == np::NP_REDUCE_ARGMIN)
				{
					if (s_A[tx] > s_A[tx + 128])
					{
						s_A[tx] = s_A[tx + 128];
						s_Idx[tx] = s_Idx[tx + 128];
					}
				}
				else if constexpr (OP == np::NP_REDUCE_ARGMAX)
				{
					if (s_A[tx] < s_A[tx + 128])
					{
						s_A[tx] = s_A[tx + 128];
						s_Idx[tx] = s_Idx[tx + 128];
					}
				}
			}
			__syncthreads();
		}
		if constexpr (BLOCK_SIZE > 127)
		{
			if (tx < 64)
			{
				if constexpr (OP == np::NP_REDUCE_ARGMIN)
				{
					if (s_A[tx] > s_A[tx + 64])
					{
						s_A[tx] = s_A[tx + 64];
						s_Idx[tx] = s_A[tx + 64];
					}
				}
				else if constexpr (OP == np::NP_REDUCE_ARGMAX)
				{
					if (s_A[tx] < s_A[tx + 64])
					{
						s_A[tx] = s_A[tx + 64];
						s_Idx[tx] = s_Idx[tx + 64];
					}
				}
			}
			__syncthreads();
		}

		if (tx < 32)
			kernelWarpReduceArgF<TP, OP>(s_A, s_Idx, tx);

		if (tx == 0)
		{
			outputMax[bx] = s_A[0];
			outputIdx[bx] = s_Idx[0];
		}

		A += size;
		A_idx += size;
		outputIdx += gridDim.x;
		outputMax += gridDim.x;
		__syncthreads();
	}
}

template <typename TP>
__global__ void kernelMatShuffle(TP *A, const int size, const unsigned long long seed)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	idx *= 4;
	if(idx < size){
		// Seed the random number generator
		curandState state;
		curand_init(seed, idx, 0, &state); // Initialize curand state for each thread
		
		int j = curand_uniform(&state) * idx; // generate random number b/w [0, idx]

		// Swap array[i] and array[j]
		TP temp = A[idx];
		A[idx] = A[j];
		A[j] = temp;

		++idx;
		if(idx < size){
			j = curand_uniform(&state) * idx; // generate random number b/w [0, idx]
			temp = A[idx];
			A[idx] = A[j];
			A[j] = temp;
		}
		++idx;
		if(idx < size){
			j = curand_uniform(&state) * idx; // generate random number b/w [0, idx]
			temp = A[idx];
			A[idx] = A[j];
			A[j] = temp;
		}
		++idx;
		if(idx < size){
			j = curand_uniform(&state) * idx; // generate random number b/w [0, idx]
			temp = A[idx];
			A[idx] = A[j];
			A[j] = temp;
		}
	}
}

template <typename TP>
__global__ void kernelReLUBackward(TP *dout, TP *cache, TP *dX, int N){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int grid_size = blockDim.x * gridDim.x;

	while(idx < N){
		dX[idx] = dout[idx] * (cache[idx] > 0);
		idx += grid_size;
	}
}

// scores = scores + 1e-8; // epsilon to prevent -log(0)
// auto loss = (-np::log(scores.get(np::arange<int>(x.rows()), y)));
// dx.set(np::arange<int>(x.rows()), y, NP_OP_SUB, 1);
template <typename TP>
__global__ void kernelSoftMaxUtils(TP *scores, const int *y, float *scores_for_loss, const int NUM_CLASSES, const int N){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int grid_size = blockDim.x * gridDim.x;

	while(idx < N){
		scores_for_loss[idx] = -(logf(scores[idx * NUM_CLASSES + y[idx]] + 1e-8));
		scores[idx * NUM_CLASSES + y[idx]] -= 1;
		idx += grid_size;
	}
}

#endif