#ifndef CUSTOMKERNELS_CUH
#define CUSTOMKERNELS_CUH

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

// kernel to set values at a range of indexes.
template <typename TP>
__global__ void kernelSetMatValues(TP *in, const int rdin, const TP *val, const int *rows, const int *cols, const int size);

// kernel to set values at a range of indexes.
template <typename TP>
__global__ void kernelSetMatValues(TP *in, const TP *val, const int *idxs, const int size);

/* Operator functions
1 for +
2 for -
3 for *
4 for /
5 for <
6 for <=
7 for >
8 for >=
9 for ==
10 for !=
*/

// operator functions

// perform operator on corrosponding elements of 2 matrices
//  C = A op B
template <typename TP, char OP>
__global__ void kernelMatOpMat(const TP *A, const TP *B, TP *C, const int size);
template <typename TP, char OP>
__global__ void kernelScalarOpMat(const TP Scal, const TP *A, TP *C, const int size);
// operator on matrix and a scalar.
//  C = A op Scal (broadcasting)
// op scalar to all values of the matrix
template <typename TP, char OP>
__global__ void kernelMatOpScalar(const TP *A, const TP Scal, TP *C, const int size);
template <typename TP, char OP>
__global__ void kernelScalarOpMat(const TP Scal, const TP *A, TP *C, const int size);

// op on matrix and vector, mat.rows = vec.dim
// C = A op V (broadcasting)
//  shapeA = M x N matrix
template <typename TP, char OP>
__global__ void kernelMatOpVecAlongCols(const TP *A, const TP *V, TP *C, const int size, const int N);
template <typename TP, char OP>
__global__ void kernelVecOpMatAlongCols(const TP *V, const TP *A, TP *C, const int size, const int N);

// operator on matrix and vector, mat.cols = vec.dim
// C = A op V (broadcasting)
// shapeA = M x N matrix
template <typename TP, char OP>
__global__ void kernelMatOpVecAlongRows(const TP *A, const TP *V, TP *C, const int size, const int N);
template <typename TP, char OP>
__global__ void kernelVecOpMatAlongRows(const TP *V, const TP *A, TP *C, const int size, const int N);

// maxmin
/*
	F
	1: min
	2: max
*/
// compare 2 matrix ( element wise ) and put max / min value in result matrix.
// A = MxN
// B = MxN
// Ci = max(Ai, Bi). (elementwise)
template <typename TP, char F>
__global__ void kernelMatMaxminMat(const TP *A, const TP *B, TP *C, const int size);

template <typename TP, char F>
__global__ void kernelMatMaxminScalar(const TP *A, const TP B, TP *C, const int size);

template <typename TP, char F>
__global__ void kernelMatMaxminVecAlongCols(const TP *A, const TP *V, TP *C, const int size, const int N);

template <typename TP, char F>
__global__ void kernelMatMaxminVecAlongRows(const TP *A, const TP *V, TP *C, const int size, const int N);

// npfunctions
// functions per element
/*
	F
	1. exp
	2. log
	3. sqaure
	4. sqrt
*/
// A = MxN
// C = MxN
// Ci = F(Ai)
template <typename TP, char F>
__global__ void kernelFMat(const TP *A, TP *C, const int size);

// np.pow
// A = MxN
// C = MxN
// Ci = square(Ai)
template <typename TP>
__global__ void kernelPowMat(const TP *A, const float power, TP *C, const int size);

// REDUCTION
/*
	F
	1: sum
	2: min
	3: max
*/
// warp unroll
template <typename TP, char F>
__device__ void kernelWarpReduceF(volatile TP *s_A, const int tid);

template <typename TP, int BLOCK_SIZE, char F>
__global__ void kernelReduceF(const TP *A, TP *output, const int size);

// warp unroll
template <typename TP, char F>
__device__ void kernelWarpReduceArgF(volatile TP *s_A, volatile int *s_Idx, const int tid);

template <typename TP, int BLOCK_SIZE, char F>
__global__ void kernelReduceArgF(const TP *A, TP *outputMax, int *outputIdx, const int size);

// second reduction k time p -> idx serial nhi h, to ek idx ka bhi array dena hoga
template <typename TP, int BLOCK_SIZE, char F>
__global__ void kernelReduceArgF(const TP *A, const int *A_idx, TP *outputMax, int *outputIdx, const int size);

// np.shuffle
template <typename TP, int BLOCK_SIZE>
__global__ void kernelMatShuffle(TP *A, const int size);

// ########## FUNCTION DEFINITIONS

// for uniform distribution
template <typename TP>
__global__ void kernelInitializeRandomUnif(TP *arr, const int size, const unsigned long long seed)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		curandState state;
		curand_init(seed, idx, 0, &state); // Initialize curand state for each thread
		arr[idx] = curand_uniform(&state); // Generate a random value
	}
}

// for uniform distribution
template <typename TP>
__global__ void kernelInitializeRandomUnif(TP *arr, const int size, const int lo, const int hi, const unsigned long long seed)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		curandState state;
		curand_init(seed, idx, 0, &state);					  // Initialize curand state for each thread
		arr[idx] = (curand_uniform(&state) * (hi - lo) + lo); // Generate a random value
	}
}

// for normal distribution
template <typename TP>
__global__ void kernelInitializeRandomNorm(TP *arr, const int size, const unsigned long long seed)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		curandState state;
		curand_init(seed, idx, 0, &state); // Initialize curand state for each thread
		arr[idx] = curand_normal(&state);  // Generate a random value
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

	if (idx < size)
	{
		in[idx] = Val;
	}
}

// kernel to initialised arange array -> [0, range), spaced by 1
template <typename TP>
__global__ void kernelInitMatArange(TP *in, const int range)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < range)
	{
		in[idx] = idx;
	}
}

// kernel to get values at a range of indexes.
template <typename TP>
__global__ void kernelGetMatValues(const TP *in, TP *out, const int *idxs, const int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		out[idx] = in[idxs[idx]];
	}
}

// kernel to get values at a range of indexes.
template <typename TP>
__global__ void kernelGetMatValues(const TP *in, const int rdin, TP *out, const int *rows, const int *cols, const int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		out[idx] = in[rows[idx] * rdin + cols[idx]];
	}
}

// kernel to set values at a range of indexes.
template <typename TP>
__global__ void kernelSetMatValues(TP *in, const int rdin, const TP *val, const int *rows, const int *cols, const int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		in[rows[idx] * rdin + cols[idx]] = val[idx];
	}
}

// kernel to set values at a range of indexes.
template <typename TP>
__global__ void kernelSetMatValues(TP *in, const TP *val, const int *idxs, const int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		in[idxs[idx]] = val[idx];
	}
}

/* Operator functions
1 for +
2 for -
3 for *
4 for /
5 for <
6 for <=
7 for >
8 for >=
9 for ==
10 for !=
*/

// operator functions

// perform operator on corrosponding elements of 2 matrices
//  C = A op B
template <typename TP, char OP>
__global__ void kernelMatOpMat(const TP *A, const TP *B, TP *C, const int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		if constexpr (OP == 1)
			C[idx] = A[idx] + B[idx];
		else if constexpr (OP == 2)
			C[idx] = A[idx] - B[idx];
		else if constexpr (OP == 3)
			C[idx] = A[idx] * B[idx];
		else if constexpr (OP == 4)
			C[idx] = A[idx] / B[idx];
		else if constexpr (OP == 5)
			C[idx] = A[idx] < B[idx];
		else if constexpr (OP == 6)
			C[idx] = A[idx] <= B[idx];
		else if constexpr (OP == 7)
			C[idx] = A[idx] > B[idx];
		else if constexpr (OP == 8)
			C[idx] = A[idx] >= B[idx];
		else if constexpr (OP == 9)
			C[idx] = A[idx] == B[idx];
		else if constexpr (OP == 10)
			C[idx] = A[idx] != B[idx];
		else
			printf("ERROR! INVALID OPERATOR IN kernelMatOPMat.\n");
	}
}

// operator on matrix and a scalar.
//  C = A op Scal (broadcasting)
// op scalar to all values of the matrix
template <typename TP, char OP>
__global__ void kernelMatOpScalar(const TP *A, const TP Scal, TP *C, const int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		if constexpr (OP == 1)
			C[idx] = A[idx] + Scal;
		else if constexpr (OP == 2)
			C[idx] = A[idx] - Scal;
		else if constexpr (OP == 3)
			C[idx] = A[idx] * Scal;
		else if constexpr (OP == 4)
			C[idx] = A[idx] / Scal;
		else if constexpr (OP == 5)
			C[idx] = A[idx] < Scal;
		else if constexpr (OP == 6)
			C[idx] = A[idx] <= Scal;
		else if constexpr (OP == 7)
			C[idx] = A[idx] > Scal;
		else if constexpr (OP == 8)
			C[idx] = A[idx] >= Scal;
		else if constexpr (OP == 9)
			C[idx] = A[idx] == Scal;
		else if constexpr (OP == 10)
			C[idx] = A[idx] != Scal;
		else
			printf("ERROR! INVALID OPERATOR IN kernelMatOPScalar.\n");
	}
}
template <typename TP, char OP>
__global__ void kernelScalarOpMat(const TP Scal, const TP *A, TP *C, const int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		if constexpr (OP == 1)
			C[idx] = Scal + A[idx];
		else if constexpr (OP == 2)
			C[idx] = Scal - A[idx];
		else if constexpr (OP == 3)
			C[idx] = Scal * A[idx];
		else if constexpr (OP == 4)
			C[idx] = Scal / A[idx];
		else if constexpr (OP == 5)
			C[idx] = Scal < A[idx];
		else if constexpr (OP == 6)
			C[idx] = Scal <= A[idx];
		else if constexpr (OP == 7)
			C[idx] = Scal > A[idx];
		else if constexpr (OP == 8)
			C[idx] = Scal >= A[idx];
		else if constexpr (OP == 9)
			C[idx] = Scal == A[idx];
		else if constexpr (OP == 10)
			C[idx] = Scal != A[idx];
		else
			printf("ERROR! INVALID OPERATOR IN kernelScalarOpMat.\n");
	}
}

// op on matrix and vector, mat.rows = vec.dim
// C = A op V (broadcasting)
//  shapeA = M x N matrix
template <typename TP, char OP>
__global__ void kernelMatOpVecAlongCols(const TP *A, const TP *V, TP *C, const int size, const int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int r = idx / N;
	// int c = idx % N;

	if (idx < size)
	{
		if constexpr (OP == 1)
			C[idx] = A[idx] + V[r];
		else if constexpr (OP == 2)
			C[idx] = A[idx] - V[r];
		else if constexpr (OP == 3)
			C[idx] = A[idx] * V[r];
		else if constexpr (OP == 4)
			C[idx] = A[idx] / V[r];
		else if constexpr (OP == 5)
			C[idx] = A[idx] < V[r];
		else if constexpr (OP == 6)
			C[idx] = A[idx] <= V[r];
		else if constexpr (OP == 7)
			C[idx] = A[idx] > V[r];
		else if constexpr (OP == 8)
			C[idx] = A[idx] >= V[r];
		else if constexpr (OP == 9)
			C[idx] = A[idx] == V[r];
		else if constexpr (OP == 10)
			C[idx] = A[idx] != V[r];
		else
			printf("ERROR! INVALID OPERATOR IN kernelScalarOpMat.\n");
	}
}
template <typename TP, char OP>
__global__ void kernelVecOpMatAlongCols(const TP *V, const TP *A, TP *C, const int size, const int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int r = idx / N;
	// int c = idx % N;

	if (idx < size)
	{
		if constexpr (OP == 1)
			C[idx] = V[r] + A[idx];
		else if constexpr (OP == 2)
			C[idx] = V[r] - A[idx];
		else if constexpr (OP == 3)
			C[idx] = V[r] * A[idx];
		else if constexpr (OP == 4)
			C[idx] = V[r] / A[idx];
		else if constexpr (OP == 5)
			C[idx] = V[r] < A[idx];
		else if constexpr (OP == 6)
			C[idx] = V[r] <= A[idx];
		else if constexpr (OP == 7)
			C[idx] = V[r] > A[idx];
		else if constexpr (OP == 8)
			C[idx] = V[r] >= A[idx];
		else if constexpr (OP == 9)
			C[idx] = V[r] == A[idx];
		else if constexpr (OP == 10)
			C[idx] = V[r] != A[idx];
		else
			printf("ERROR! INVALID OPERATOR IN kernelScalarOpMat.\n");
	}
}

// operator on matrix and vector, mat.cols = vec.dim
// C = A op V (broadcasting)
// shapeA = M x N matrix
template <typename TP, char OP>
__global__ void kernelMatOpVecAlongRows(const TP *A, const TP *V, TP *C, const int size, const int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// int r = idx / N;
	int c = idx % N;

	if (idx < size)
	{
		if constexpr (OP == 1)
			C[idx] = A[idx] + V[c];
		else if constexpr (OP == 2)
			C[idx] = A[idx] - V[c];
		else if constexpr (OP == 3)
			C[idx] = A[idx] * V[c];
		else if constexpr (OP == 4)
			C[idx] = A[idx] / V[c];
		else if constexpr (OP == 5)
			C[idx] = A[idx] < V[c];
		else if constexpr (OP == 6)
			C[idx] = A[idx] <= V[c];
		else if constexpr (OP == 7)
			C[idx] = A[idx] > V[c];
		else if constexpr (OP == 8)
			C[idx] = A[idx] >= V[c];
		else if constexpr (OP == 9)
			C[idx] = A[idx] == V[c];
		else if constexpr (OP == 10)
			C[idx] = A[idx] != V[c];
		else
			printf("ERROR! INVALID OPERATOR IN kernelScalarOpMat.\n");
	}
}
template <typename TP, char OP>
__global__ void kernelVecOpMatAlongRows(const TP *V, const TP *A, TP *C, const int size, const int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// int r = idx / N;
	int c = idx % N;

	if (idx < size)
	{
		if constexpr (OP == 1)
			C[idx] = V[c] + A[idx];
		else if constexpr (OP == 2)
			C[idx] = V[c] - A[idx];
		else if constexpr (OP == 3)
			C[idx] = V[c] * A[idx];
		else if constexpr (OP == 4)
			C[idx] = V[c] / A[idx];
		else if constexpr (OP == 5)
			C[idx] = V[c] < A[idx];
		else if constexpr (OP == 6)
			C[idx] = V[c] <= A[idx];
		else if constexpr (OP == 7)
			C[idx] = V[c] > A[idx];
		else if constexpr (OP == 8)
			C[idx] = V[c] >= A[idx];
		else if constexpr (OP == 9)
			C[idx] = V[c] == A[idx];
		else if constexpr (OP == 10)
			C[idx] = V[c] != A[idx];
		else
			printf("ERROR! INVALID OPERATOR IN kernelScalarOpMat.\n");
	}
}

// maxmin
/*
	F
	1: min
	2: max
*/
// compare 2 matrix ( element wise ) and put max / min value in result matrix.
// A = MxN
// B = MxN
// Ci = F(Ai, Bi). (elementwise)
template <typename TP, char F>
__global__ void kernelMatMaxminMat(const TP *A, const TP *B, TP *C, const int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		if constexpr (F == 1)
			C[idx] = (A[idx] < B[idx]) ? A[idx] : B[idx];
		else if constexpr (F == 2)
			C[idx] = (A[idx] > B[idx]) ? A[idx] : B[idx];
		else
			printf("ERROR! INVALID OPERATOR IN kernelMatMaxminMat.\n");
	}
}

// max/min of matrix elements and a scalar.
//  Ci = max(Ai, Scal) (broadcasting)
template <typename TP, char F>
__global__ void kernelMatMaxminScalar(const TP *A, const TP Scal, TP *C, const int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		if constexpr (F == 1)
			C[idx] = (A[idx] < Scal) ? A[idx] : Scal;
		else if constexpr (F == 2)
			C[idx] = (A[idx] > Scal) ? A[idx] : Scal;
		else
			printf("ERROR! INVALID OPERATOR IN kernelMatMaxminScalar.\n");
	}
}
// max/min of matrix elements and a vector. vec.dim = mat.rows
//  Ci = max(Ai, Vr) (broadcasting)
//  shapeA = M x N matrix
template <typename TP, char F>
__global__ void kernelMatMaxminVecAlongCols(const TP *A, const TP *V, TP *C, const int size, const int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int r = idx / N;
	// int c = idx % N;

	if (idx < size)
	{
		if constexpr (F == 1)
			C[idx] = (A[idx] < V[r]) ? A[idx] : V[r];
		else if constexpr (F == 2)
			C[idx] = (A[idx] > V[r]) ? A[idx] : V[r];
		else
			printf("ERROR! INVALID OPERATOR IN kernelMatMaxmiVecAlongCols.\n");
	}
}

// max/min of matrix elements and a vector. vec.dim = mat.cols
//  Ci = max(Ai, Vc) (broadcasting)
//  shapeA = M x N matrix
template <typename TP, char F>
__global__ void kernelMatMaxminVecAlongRows(const TP *A, const TP *V, TP *C, const int size, const int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// int r = idx / N;
	int c = idx % N;

	if (idx < size)
	{
		if constexpr (F == 1)
			C[idx] = (A[idx] < V[c]) ? A[idx] : V[c];
		else if constexpr (F == 2)
			C[idx] = (A[idx] > V[c]) ? A[idx] : V[c];
		else
			printf("ERROR! INVALID OPERATOR IN kernelMatMaxminVecAlongRows.\n");
	}
}

// npfunctions
// functions per element
/*
	F
	1. exp
	2. log
	3. sqaure
	4. sqrt
*/
// A = MxN
// C = MxN
// Ci = F(Ai)
template <typename TP, char F>
__global__ void kernelFMat(const TP *A, TP *C, const int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		if constexpr (F == 1)
			C[idx] = expf(A[idx]);
		else if constexpr (F == 2)
			C[idx] = logf(A[idx]);
		else if constexpr (F == 3)
			C[idx] = A[idx] * A[idx];
		else if constexpr (F == 4)
		{
			if constexpr (std::is_same<TP, int>::value)
				C[idx] = static_cast<int>(sqrtf(A[idx]));
			else if constexpr (std::is_same<TP, float>::value)
				C[idx] = sqrtf(A[idx]);
			else if constexpr (std::is_same<TP, double>::value)
				C[idx] = sqrt(A[idx]);
			else
				// Handle other types here
				printf("Unsupported type in kernelFMat, where F = 4(sqrt)");
		}
		else
			printf("Unsupported function type in kernelFMat\n");
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

	if (idx < size)
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
	}
}

// REDUCTION

// REDUCTION
/*
	F
	1: sum
	2: min
	3: max
*/
template <typename TP, char F>
__device__ void kernelWarpReduceF(volatile TP *s_A, const int tid)
{ // warp reduce for kernel
	if constexpr (F == 1)
	{
		s_A[tid] += s_A[tid + 32];
		s_A[tid] += s_A[tid + 16];
		s_A[tid] += s_A[tid + 8];
		s_A[tid] += s_A[tid + 4];
		s_A[tid] += s_A[tid + 2];
		s_A[tid] += s_A[tid + 1];
	}
	else if constexpr (F == 2)
	{
		s_A[tid] = min(s_A[tid], s_A[tid + 32]);
		s_A[tid] = min(s_A[tid], s_A[tid + 16]);
		s_A[tid] = min(s_A[tid], s_A[tid + 8]);
		s_A[tid] = min(s_A[tid], s_A[tid + 4]);
		s_A[tid] = min(s_A[tid], s_A[tid + 2]);
		s_A[tid] = min(s_A[tid], s_A[tid + 1]);
	}
	else if constexpr (F == 3)
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
template <typename TP, int BLOCK_SIZE, char F>
__global__ void kernelReduceF(const TP *A, TP *output, const int size)
{
	if constexpr (F != 1 && F != 2 && F != 3)
	{
		printf("INVALID ARGUMENT! in kernelReduceF\n");
		return;
	}
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	int idx = bx * BLOCK_SIZE * 2 + tx;
	const int gridSize = BLOCK_SIZE * 2 * gridDim.x;
	__shared__ TP s_A[BLOCK_SIZE];

	if constexpr (F == 1)
		s_A[tx] = 0;
	else if constexpr (F == 2)
		s_A[tx] = INT_MAX;
	else if constexpr (F == 3)
		s_A[tx] = INT_MIN;
	// assume 1 hi grid launch kr rha h tu
	while (idx < size)
	{
		if constexpr (F == 1)
			s_A[tx] += (A[idx] + ((idx + BLOCK_SIZE < size) ? A[idx + BLOCK_SIZE] : 0));
		else if constexpr (F == 2)
		{
			s_A[tx] = min(s_A[tx], A[idx]);
			if (idx + BLOCK_SIZE < size)
				s_A[tx] = min(s_A[tx], A[idx + BLOCK_SIZE]);
		}
		else if constexpr (F == 3)
		{
			s_A[tx] = max(s_A[tx], A[idx]);
			if (idx + BLOCK_SIZE < size)
				s_A[tx] = max(s_A[tx], A[idx + BLOCK_SIZE]);
		}

		idx += gridSize;
	}
	__syncthreads();

	if constexpr (BLOCK_SIZE > 511)
	{
		if (tx < 256)
		{
			if constexpr (F == 1)
				s_A[tx] += s_A[tx + 256];
			else if constexpr (F == 2)
				s_A[tx] = min(s_A[tx], s_A[tx + 256]);
			else if constexpr (F == 3)
				s_A[tx] = max(s_A[tx], s_A[tx + 256]);
		}
		__syncthreads();
	}

	if constexpr (BLOCK_SIZE > 255)
	{
		if (tx < 128)
		{
			if constexpr (F == 1)
				s_A[tx] += s_A[tx + 128];
			else if constexpr (F == 2)
				s_A[tx] = min(s_A[tx], s_A[tx + 128]);
			else if constexpr (F == 3)
				s_A[tx] = max(s_A[tx], s_A[tx + 128]);
		}
		__syncthreads();
	}
	if constexpr (BLOCK_SIZE > 127)
	{
		if (tx < 64)
		{
			if constexpr (F == 1)
				s_A[tx] += s_A[tx + 64];
			else if constexpr (F == 2)
				s_A[tx] = min(s_A[tx], s_A[tx + 64]);
			else if constexpr (F == 3)
				s_A[tx] = max(s_A[tx], s_A[tx + 64]);
		}
		__syncthreads();
	}

	if (tx < 32)
		kernelWarpReduceF<TP, F>(s_A, tx);

	if (tx == 0)
		output[bx] = s_A[0];
}

// ArgReduction
/*
	F
	2: min
	3: max
*/
template <typename TP, char F>
__device__ void kernelWarpReduceArgF(volatile TP *s_A, volatile int *s_Idx, const int tid)
{ // warp reduce for kernel
	if constexpr (F == 2)
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
	else if constexpr (F == 3)
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
template <typename TP, int BLOCK_SIZE, char F>
__global__ void kernelReduceArgF(const TP *A, TP *outputMax, int *outputIdx, const int size)
{
	if constexpr (F != 2 && F != 3)
	{
		printf("INVALID ARGUMENT! in kernelReduceArgF\n");
		return;
	}
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	int idx = bx * BLOCK_SIZE * 2 + tx;
	const int gridSize = BLOCK_SIZE * 2 * gridDim.x;
	__shared__ TP s_A[BLOCK_SIZE];
	__shared__ int s_Idx[BLOCK_SIZE];

	if constexpr (F == 2)
		s_A[tx] = INT_MAX;
	else if constexpr (F == 3)
		s_A[tx] = INT_MIN;
	s_A[tx] = -1;

	// assume 1 hi grid launch kr rha h tu
	while (idx < size)
	{
		if constexpr (F == 2)
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
		else if constexpr (F == 3)
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

		idx += gridSize;
	}
	__syncthreads();

	if constexpr (BLOCK_SIZE > 511)
	{
		if (tx < 256)
		{
			if constexpr (F == 2)
			{
				if (s_A[tx] > s_A[idx + 256])
				{
					s_A[tx] = s_A[idx + 256];
					s_Idx[tx] = idx + 256;
				}
			}
			else if constexpr (F == 3)
			{
				if (s_A[tx] < s_A[idx + 256])
				{
					s_A[tx] = s_A[idx + 256];
					s_Idx[tx] = idx + 256;
				}
			}
		}
		__syncthreads();
	}

	if constexpr (BLOCK_SIZE > 255)
	{
		if (tx < 128)
		{
			if constexpr (F == 2)
			{
				if (s_A[tx] > s_A[idx + 128])
				{
					s_A[tx] = s_A[idx + 128];
					s_Idx[tx] = idx + 128;
				}
			}
			else if constexpr (F == 3)
			{
				if (s_A[tx] < s_A[idx + 128])
				{
					s_A[tx] = s_A[idx + 128];
					s_Idx[tx] = idx + 128;
				}
			}
		}
		__syncthreads();
	}
	if constexpr (BLOCK_SIZE > 127)
	{
		if (tx < 64)
		{
			if constexpr (F == 2)
			{
				if (s_A[tx] > s_A[idx + 64])
				{
					s_A[tx] = s_A[idx + 64];
					s_Idx[tx] = idx + 64;
				}
			}
			else if constexpr (F == 3)
			{
				if (s_A[tx] < s_A[idx + 64])
				{
					s_A[tx] = s_A[idx + 64];
					s_Idx[tx] = idx + 64;
				}
			}
		}
		__syncthreads();
	}

	if (tx < 32)
		kernelWarpReduceArgF<TP, F>(s_A, s_Idx, tx);

	if (tx == 0)
	{
		outputMax[bx] = s_A[0];
		outputIdx[bx] = s_Idx[0];
	}
}

// second reduction k time p -> idx serial nhi h, to ek idx ka bhi array dena hoga
template <typename TP, int BLOCK_SIZE, char F>
__global__ void kernelReduceArgF(const TP *A, const int *A_idx, TP *outputMax, int *outputIdx, const int size)
{
	if constexpr (F != 2 && F != 3)
	{
		printf("INVALID ARGUMENT! in kernelReduceArgF\n");
		return;
	}
	const int bx = blockIdx.x;
	const int tx = threadIdx.x;
	int idx = bx * BLOCK_SIZE * 2 + tx;
	const int gridSize = BLOCK_SIZE * 2 * gridDim.x;
	__shared__ TP s_A[BLOCK_SIZE];
	__shared__ int s_Idx[BLOCK_SIZE];

	if constexpr (F == 2)
		s_A[tx] = INT_MAX;
	else if constexpr (F == 3)
		s_A[tx] = INT_MIN;
	s_A[tx] = -1;

	// assume 1 hi grid launch kr rha h tu
	while (idx < size)
	{
		if constexpr (F == 2)
		{
			if (s_A[tx] > A[idx])
			{
				s_A[tx] = A[idx];
				s_Idx[tx] = A_idx[idx];
			}
			if (idx + BLOCK_SIZE < size)
			{
				if (s_A[tx] > A[idx + BLOCK_SIZE])
				{
					s_A[tx] = A[idx + BLOCK_SIZE];
					s_Idx[tx] = A_idx[idx + BLOCK_SIZE];
				}
			}
		}
		else if constexpr (F == 3)
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

		idx += gridSize;
	}
	__syncthreads();

	if constexpr (BLOCK_SIZE > 511)
	{
		if (tx < 256)
		{
			if constexpr (F == 2)
			{
				if (s_A[tx] > s_A[idx + 256])
				{
					s_A[tx] = s_A[idx + 256];
					s_Idx[tx] = idx + 256;
				}
			}
			else if constexpr (F == 3)
			{
				if (s_A[tx] < s_A[idx + 256])
				{
					s_A[tx] = s_A[idx + 256];
					s_Idx[tx] = idx + 256;
				}
			}
		}
		__syncthreads();
	}

	if constexpr (BLOCK_SIZE > 255)
	{
		if (tx < 128)
		{
			if constexpr (F == 2)
			{
				if (s_A[tx] > s_A[idx + 128])
				{
					s_A[tx] = s_A[idx + 128];
					s_Idx[tx] = idx + 128;
				}
			}
			else if constexpr (F == 3)
			{
				if (s_A[tx] < s_A[idx + 128])
				{
					s_A[tx] = s_A[idx + 128];
					s_Idx[tx] = idx + 128;
				}
			}
		}
		__syncthreads();
	}
	if constexpr (BLOCK_SIZE > 127)
	{
		if (tx < 64)
		{
			if constexpr (F == 2)
			{
				if (s_A[tx] > s_A[idx + 64])
				{
					s_A[tx] = s_A[idx + 64];
					s_Idx[tx] = idx + 64;
				}
			}
			else if constexpr (F == 3)
			{
				if (s_A[tx] < s_A[idx + 64])
				{
					s_A[tx] = s_A[idx + 64];
					s_Idx[tx] = idx + 64;
				}
			}
		}
		__syncthreads();
	}

	if (tx < 32)
		kernelWarpReduceArgF<TP, F>(s_A, s_Idx, tx);

	if (tx == 0)
	{
		outputMax[bx] = s_A[0];
		outputIdx[bx] = s_Idx[0];
	}
}

template <typename TP>
__global__ void kernelMatShuffle(TP *A, const int size, const unsigned long long seed)
{
	if (size <= 1)
		; // No need to shuffle if size is 0 or 1
	else
	{
		// Seed the random number generator
		curandState state;
		curand_init(seed, 0, 0, &state); // Initialize curand state for each thread

		for (int i = size - 1; i > 0; --i)
		{
			// Generate a random index between 0 and i (inclusive)
			int j = curand_uniform(&state) * i;

			// Swap array[i] and array[j]
			TP temp = A[i];
			A[i] = A[j];
			A[j] = temp;
		}
	}
}

#endif