#ifndef NPGPUARRAY_CUH
#define NPGPUARRAY_CUH

#include <numC/npGPUArray.cuh>
#include <numC/customKernels.cuh>
#include <numC/errorCheckUtils.cuh>
#include <numC/gpuConfig.cuh>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>
#include <vector>


namespace np
{
	#define ceil(x, y) ((x + y - 1) / y)
	#define NP_OP_ADD 1
	#define NP_OP_SUB 2
	#define NP_OP_MUL 3
	#define NP_OP_DIV 4
	#define NP_OP_LESS_THAN 5
	#define NP_OP_LESS_THAN_EQ 6
	#define NP_OP_GREATER_THAN 7
	#define NP_OP_GREATER_THAN_EQ 8
	#define NP_OP_EQ 9
	#define NP_OP_NOT_EQ 10
	#define NP_OP_MINIMUM 11
	#define NP_OP_MAXIMUM 12

	#define NP_SET_EQ 13

	#define NP_REDUCE_SUM 14
	#define NP_REDUCE_MIN 15
	#define NP_REDUCE_MAX 16
	#define NP_REDUCE_ARGMIN 17
	#define NP_REDUCE_ARGMAX 18

	#define NP_F_EXP 19
	#define NP_F_LOG 20
	#define NP_F_SQAURE 21
	#define NP_F_SQRT 22
	#define NP_F_POW 23

	template <typename TP>
	class ArrayGPU
	{
	private:
		// reference counter -> Implemented so that all arrays can share the same memory. array will be deleted when this becomes 0.
		int *ref_count; 
		// _rows and _cols of array.
		int _rows, _cols;
		
		template <char OP>
		ArrayGPU<TP> applyOp(const ArrayGPU<TP> &B) const;
		template <char OP>
		ArrayGPU<TP> applyOp(const TP Scalar) const;

		template <char F>
		ArrayGPU<TP> applyReductionF(const int axis) const;
		template <char F>
		ArrayGPU<int> applyReductionArgF(const int axis) const;

	public:
		// main array.
		TP *mat;
		friend class Random;

		// ####################### CONSTRUCTORS ############################

		/*
			Default constructor
			Sets _rows = _cols = 0
		*/
		ArrayGPU();

		/*
			Parameterised constructor
			Creates a 1D array
			Arguments:
			* sz = size of array 
			Ex: ArrayGPU<float>(10);
		*/
		ArrayGPU(const int sz);

		/*
			Parameterised constructor
			Creates a 2D array
			Arguments:
			* _rows = _rows in array
			* _cols = _cols in array
			Ex: ArrayGPU<float>(3, 4);
		*/
		ArrayGPU(const int _rows, const int _cols);

		/*
			Parameterised constructor
			Creates a 2D array fill with a default value.
			Arguments:
			* _rows = _rows in array
			* _cols = _cols in array
			* Val = Scalar to fill the array with
			Ex: ArrayGPU<float>(3, 4, 0);
		*/
		ArrayGPU(const int _rows, const int _cols, const TP Val);

		/*
			Parameterised constructor
			Creates a 1D array with values taking from std::vector.
			Arguments:
			* std::vector<>
			Ex: ArrayGPU<float>({0, 1, 2, 3, 4, 5});
		*/
		ArrayGPU(const std::vector<TP> &A);

		/*
			Parameterised constructor
			Creates a 2D array with values taking from std::vector.
			Arguments:
			* std::vector<std::vector<>>
			Ex: ArrayGPU<float>({{0, 1, 2, 3, 4, 5},
								 {6, 7, 8, 9, 10, 11});
		*/
		ArrayGPU(const std::vector<std::vector<TP>> &A);

		/*
			Parameterised constructor
			Creates a 1D array and copies data from a pointer to array in memory
			Arguments:
			* TP* 
			* sz = size of array
			* loc = "cpu" or "gpu"
			Ex: ArrayGPU<float>(array, 5, "cpu");
		*/
		ArrayGPU(const TP *h_array, const int sz, const std::string &loc = "cpu");

		/*
			Parameterised constructor
			Creates a 2D array and copies data from a pointer to array in memory
			Arguments:
			* TP* 
			* _rows = _rows of mat
			* _cols = _cols of mat
			* loc = "cpu" or "gpu"
			Ex: ArrayGPU<float>(array, 5, 6, "gpu");
		*/
		ArrayGPU(const TP *h_array, int _rows, const int _cols, const std::string &loc = "cpu"); 

		/*
			Copy constructor
		*/
		ArrayGPU(const ArrayGPU<TP> &A);

		/*
			assignment operator overload
		*/
		void operator=(const ArrayGPU<TP> &A);

		/*
			assignment operator overload. this fills the array with Scal value.
			created to be used with indexing
		*/
		void operator=(const TP Scal);

		// ####################### GETTER FUNCTIONS ############################

		/*
			returns size of array.
			Ex: A.size();
		*/
		const unsigned int size() const;
	
		/*
			returns _rows of array.
			Ex: A._rows();
		*/
		const unsigned int rows() const;

		/*
			returns _cols of array.
			Ex: A._cols();
		*/
		const unsigned int cols() const;

		/*
			returns reference count of array.
			Ex: A.refCount();
		*/
		const unsigned int refCount() const;

		// ####################### ARRAY UTILITY FUNCTIONS ############################

		/*
			Prints the array on stdout.
			Ex: A.print();
		*/
		void print() const;
		
		/*
			Overloaded cout
			Ex: std::cout<<A;
		*/
		// friend std::ostream &operator<<(std::ostream &out, const ArrayGPU<TP> &A);

		/*
			Returns a copy of array.
			Ex: auto B = A.copy();
		*/
		ArrayGPU<TP> copy() const;

		/*
			Returns transpose of array.
			Ex: auto AT = A.T();
		*/
		ArrayGPU<TP> T() const;
		
		/*
			Reshapes the array. Org size has to be same as new size.
			Arguments:
			* newRows - number of _rows
			* newCols - number of _cols
			Ex: A.reshape(5, 10);
		*/
		void reshape(const int newRows, const int newCols);

		/*
			Returns a copy of array as cpu pointer.
			Ex: float *a = A.cpu();
		*/
		TP* cpu() const;
		
		// ####################### ARRAY ELEMENT GETTER SETTER FUNCTIONS ############################

		/*
			Returns element at idx as array object.
			Arguments:
			* idx - idx of element
			Ex: auto B = A.at(0);
		*/
		ArrayGPU<TP> at(const int idx) const;

		/*
			Returns element at (r, c) as array object.
			Arguments:
			* r - r to access from
			* c - c to access from
			Ex: auto B = A.at(5, 2);
		*/
		ArrayGPU<TP> at(const int r, const int c) const;

		// ######################### TO BE DONE #########################################
		/*
			Returns elements at std::vector<int> indexes as array object.
			Arguments:
			* std::vector<int> - indexes to fetch from
			Ex: auto B = A.at({1, 2, 3, 4, 5});
		*/
		// ArrayGPU<TP> at(const std::vector<int> &idxs) const;

		/*
			Returns elements at ArrayGPU<int> indexes as array object.
			Arguments:
			* ArrayGPU<float> - indexes to fetch from

			Ex: auto B = A.at({1, 2, 3, 4, 5});
		*/
		// ArrayGPU<TP> at(const ArrayGPU<int> &idxs) const;

		/*
			Returns elements at ArrayGPU<int> && ArrayGPU<int> indexes as array object.
			Arguments:
			* idx - idx of element
			Ex: auto B = A.at(R, C);
		*/
		// ArrayGPU<TP> at(const ArrayGPU<int> &r, const ArrayGPU<int> &c) const;

		/*
			Returns elements at std::vector<int> && ArrayGPU<int> indexes as array object.
			Arguments:
			* idx - idx of element
			Ex: auto B = A.at({1, 2, 3, 4, 5}, C);
		*/
		// ArrayGPU<TP> at(const std::vector<int> &r, const ArrayGPU<int> &c) const;

		/*
			Returns elements at ArrayGPU<int> && std::vector<int> indexes as array object.
			Arguments:
			* idx - idx of element
			Ex: auto B = A.at(R, {1, 2, 3, 4, 5});
		*/
		// ArrayGPU<TP> at(const ArrayGPU<int> &r, const std::vector<int> &c) const;

		/*
			Returns elements at std::vector<> && ArrayGPU<int> indexes as array object.
			Arguments:
			* idx - idx of element
			Ex: auto B = A.at({1, 2, 3, 4, 5}, {1, 2, 3, 4, 5, 6, 7, 8});
		*/
		// ArrayGPU<TP> at(const std::vector<int> &r, const std::vector<int> &c) const;

		// ####################### DOT PRODUCT ############################

		/*
			Returns dot product of two arrays
			Arguments:
			* B - second array
			Ex: auto C = A.dot(B);
		*/
		ArrayGPU<TP> dot(const ArrayGPU<TP> &B) const;
		/*
			Returns dot product of two arrays. First is transposed
			Arguments:
			* B - second array
			Ex: auto C = A.Tdot(B);
			Is same as - auto C = A.T().dot(B)
		*/
		ArrayGPU<TP> Tdot(const ArrayGPU<TP> &B) const;
		/*
			Returns dot product of two arrays. Second is transposed
			Arguments:
			* B - second array
			Ex: auto C = A.dotT(B);
			Is same as - auto C = A.dot(B.T())
		*/
		ArrayGPU<TP> dotT(const ArrayGPU<TP> &B) const;

		// TO BE DONE.
		

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
		// element wise comparison

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

	// ####################### CONSTRUCTORS ############################

	/*
		Default constructor
		Sets _rows = _cols = 0
	*/
	template <typename TP>
	ArrayGPU<TP>::ArrayGPU(){
		this->_rows = 0;
		this->_cols = 0;
	}

	/*
		Parameterised constructor
		Creates a 1D array
		Arguments:
		* sz = size of array 
		Ex: ArrayGPU<float>(10);
	*/
	template<typename TP>
	ArrayGPU<TP>::ArrayGPU(const int sz){
		this->_rows = 1;
		this->_cols = sz;

		CUDA_CALL(cudaMalloc((void **)&this->mat, this->_rows * this->_cols * sizeof(TP)));

		// initialising ref_count
		this->ref_count = (int*)malloc(sizeof(int));
		(*this->ref_count) = 1;
	}

	/*
		Parameterised constructor
		Creates a 2D array
		Arguments:
		* _rows = _rows in array
		* _cols = _cols in array
		Ex: ArrayGPU<float>(3, 4);
	*/
	template <typename TP>
	ArrayGPU<TP>::ArrayGPU(const int _rows, const int _cols)
	{
		this->_rows = _rows;
		this->_cols = _cols;

		CUDA_CALL(cudaMalloc((void **)&this->mat, this->_rows * this->_cols * sizeof(TP)));

		// initialising ref_count
		this->ref_count = (int*)malloc(sizeof(int));
		(*this->ref_count) = 1;
	}

	/*
		Parameterised constructor
		Creates a 2D array fill with a default value.
		Arguments:
		* _rows = _rows in array
		* _cols = _cols in array
		* Val = Scalar to fill the array with
		Ex: ArrayGPU<float>(3, 4, 0);
	*/
	template <typename TP>
	ArrayGPU<TP>::ArrayGPU(const int _rows, const int _cols, const TP Val)
	{
		this->_rows = _rows;
		this->_cols = _cols;

		CUDA_CALL(cudaMalloc((void **)&this->mat, this->_rows * this->_cols * sizeof(TP)));

		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(std::min<int>(GPU_NUM_SM * 2, ceil(this->_rows * this->_cols, block.x)));

		kernelInitMatBroadcast<TP><<<grid, block>>>(mat, Val, this->_rows * this->_cols);
		cudaDeviceSynchronize();

		// initialising ref_count
		this->ref_count = (int*)malloc(sizeof(int));
		(*this->ref_count) = 1;
	}
	
	/*
		Parameterised constructor
		Creates a 1D array with values taking from std::vector.
		Arguments:
		* std::vector<>
		Ex: ArrayGPU<float>({0, 1, 2, 3, 4, 5});
	*/
	template <typename TP>
	ArrayGPU<TP>::ArrayGPU(const std::vector<TP> &A)
	{
		this->_rows = 1;
		this->_cols = A.size();

		CUDA_CALL(cudaMalloc((void **)&this->mat, this->_rows * this->_cols * sizeof(TP)));

		CUDA_CALL(cudaMemcpy(this->mat, A.data(), this->_rows * this->_cols * sizeof(TP), cudaMemcpyHostToDevice));

		// initialising ref_count
		this->ref_count = (int*)malloc(sizeof(int));
		(*this->ref_count) = 1;
	}
	
	/*
		Parameterised constructor
		Creates a 2D array with values taking from std::vector.
		Arguments:
		* std::vector<std::vector<>>
		Ex: ArrayGPU<float>({{0, 1, 2, 3, 4, 5},
							{6, 7, 8, 9, 10, 11});
	*/
	template <typename TP>
	ArrayGPU<TP>::ArrayGPU(const std::vector<std::vector<TP>> &A)
	{
		this->_rows = A.size();
		this->_cols = A[0].size();

		CUDA_CALL(cudaMalloc((void **)&this->mat, this->_rows * this->_cols * sizeof(TP)));

		for (int rowIdx = 0; rowIdx < this->_rows; ++rowIdx)
			CUDA_CALL(cudaMemcpy(mat + rowIdx * this->_cols, A[rowIdx].data(), this->_cols * sizeof(TP), cudaMemcpyHostToDevice));

		// initialising ref_count
		this->ref_count = (int*)malloc(sizeof(int));
		(*this->ref_count) = 1;
	}
	
	/*
		Parameterised constructor
		Creates a 1D array and copies data from a pointer to array in memory
		Arguments:
		* TP* 
		* sz = size of array
		* loc = "cpu" or "gpu"
		Ex: ArrayGPU<float>(array, 5, "cpu");
	*/
	template<typename TP>
	ArrayGPU<TP>::ArrayGPU(const TP *array, const int sz, const std::string &loc){
		this->_rows = 1;
		this->_cols = sz;

		if(loc == "cpu"){
			CUDA_CALL(cudaMalloc((void**)&this->mat, this->_rows * this->_cols * sizeof(float)));
			CUDA_CALL(cudaMemcpy(this->mat, array, this->_rows * this->_cols * cudaMemcpyHostToDevice));
			// initialising ref_count
			this->ref_count = (int*)malloc(sizeof(int));
			(*this->ref_count) = 1;
		}
		else if(loc == "gpu"){
			CUDA_CALL(cudaMalloc((void**)&this->mat, this->_rows * this->_cols * sizeof(float)));
			CUDA_CALL(cudaMemcpy(this->mat, array, this->_rows * this->_cols * cudaMemcpyDeviceToDevice));
			// initialising ref_count
			this->ref_count = (int*)malloc(sizeof(int));
			(*this->ref_count) = 1;
		}
		else
			std::cerr<<"INVALID PARAM LOC: POSSIBLE VALUES \"cpu\", \"gpu\"\n";
		
	}
	
	/*
		Parameterised constructor
		Creates a 2D array and copies data from a pointer to array in memory
		Arguments:
		* TP* 
		* _rows = _rows of mat
		* _cols = _cols of mat
		* loc = "cpu" or "gpu"
		Ex: ArrayGPU<float>(array, 5, 6, "gpu");
	*/
	template<typename TP>
	ArrayGPU<TP>::ArrayGPU(const TP *array, const int _rows, const int _cols, const std::string &loc){
		this->_rows = _rows;
		this->_cols = _cols;

		if(loc == "cpu"){
			CUDA_CALL(cudaMalloc((void**)&this->mat, this->_rows * this->_cols * sizeof(TP)));
			CUDA_CALL(cudaMemcpy(this->mat, array, this->_rows * this->_cols * sizeof(TP), cudaMemcpyHostToDevice));
			// initialising ref_count
			this->ref_count = (int*)malloc(sizeof(int));
			(*this->ref_count) = 1;
		}
		else if(loc == "gpu"){
			CUDA_CALL(cudaMalloc((void**)&this->mat, this->_rows * this->_cols * sizeof(TP)));
			CUDA_CALL(cudaMemcpy(this->mat, array, this->_rows * this->_cols * sizeof(TP), cudaMemcpyDeviceToDevice));
			// initialising ref_count
			this->ref_count = (int*)malloc(sizeof(int));
			(*this->ref_count) = 1;
		}
		else
			std::cerr<<"INVALID PARAM LOC: POSSIBLE VALUES \"cpu\", \"gpu\"\n";
	} 

	/*
		Copy constructor
	*/
	template <typename TP>
	ArrayGPU<TP>::ArrayGPU(const ArrayGPU<TP> &A)
	{
		this->_rows = A._rows;
		this->_cols = A._cols;
		this->mat = A.mat;
		this->ref_count = A.ref_count;
		++(*this->ref_count);
	}

	/*
		assignment operator overload
	*/
	template <typename TP>
	void ArrayGPU<TP>::operator=(const ArrayGPU<TP> &A)
	{	
		if(this != &A){
			this->_rows = A._rows;
			this->_cols = A._cols;
			--(*this->ref_count);
			if(*this->ref_count == 0){
				cudaFree(this->mat);
				free(this->ref_count);
			}
			this->mat = A.mat;
			this->ref_count = A.ref_count;
			++(*this->ref_count);
		}
	}
	
	/*
		assignment operator overload. this fills the array with Scal value.
		created to be used with indexing
	*/
	template <typename TP>
	void ArrayGPU<TP>::operator=(const TP Scal){
		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(std::min<int>(GPU_NUM_SM * 2, ceil(this->_rows * this->_cols, block.x)));

		kernelInitMatBroadcast<TP><<<grid, block>>>(mat, Scal, this->_rows * this->_cols);
		cudaDeviceSynchronize();
	}

	// ####################### GETTER FUNCTIONS ############################

	/*
		returns size of array.
		Ex: A.size();
	*/
	template <typename TP>
	const unsigned int ArrayGPU<TP>::size() const{
		return this->_rows * this->_cols;
	}

	/*
		returns _rows of array.
		Ex: A._rows();
	*/
	template <typename TP>
	const unsigned int ArrayGPU<TP>::rows() const{
		return this->_rows;
	}

	/*
		returns _cols of array.
		Ex: A._cols();
	*/
	template <typename TP>
	const unsigned int ArrayGPU<TP>::cols() const{
		return this->_cols;
	}

	/*
		returns reference count of array.
		Ex: A.refCount();
	*/
	template <typename TP>
	const unsigned int ArrayGPU<TP>::refCount() const{
		return (*this->ref_count);
	}

	// ####################### ARRAY UTILITY FUNCTIONS ############################

	/*
		Prints the array on stdout.
		Ex: A.print();
	*/
	template <typename TP>
	void ArrayGPU<TP>::print() const
	{
		kernelPrintMat<TP><<<1, 1>>>(mat, this->_rows, this->_cols);
		cudaDeviceSynchronize();
	}

	/*
		Overloaded cout
		Ex: std::cout<<A;
	*/
	template <typename TP>
	std::ostream& operator<<(std::ostream &out, const ArrayGPU<TP> &A)
	{
		A.print();
		return out;
	}

	/*
		Returns a copy of array.
		Ex: auto B = A.copy();
	*/ 
	template<typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::copy() const {
		return ArrayGPU<TP>(this->mat, this->_rows, this->_cols, "gpu");
	}

	/*
		Returns transpose of array.
		Ex: auto AT = A.T();
	*/
	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::T() const
	{
		ArrayGPU<TP> out(this->_cols, this->_rows);
		const int TILE_WIDTH = (GPU_NUM_CUDA_CORE == 64) ? 8 : 16;
		const int ROW_BLOCK = (GPU_NUM_CUDA_CORE == 64) ? 4 : 8;
		dim3 block(TILE_WIDTH, ROW_BLOCK);
		dim3 grid(ceil(this->_cols, TILE_WIDTH), ceil(this->_rows, TILE_WIDTH));

		switch (GPU_NUM_CUDA_CORE)
		{
		case 64:
			kernelTransposeInMem<TP, 8, 4><<<grid, block>>>(this->mat, out.mat, this->_rows, this->_cols);
			break;

		default:
			kernelTransposeInMem<TP, 16, 8><<<grid, block>>>(this->mat, out.mat, this->_rows, this->_cols);
			break;
		}
		cudaDeviceSynchronize();

		return out;
	}

	/*
		Reshapes the array. Org size has to be same as new size.
		Arguments:
		* newRows - number of _rows
		* newCols - number of _cols
		Ex: A.reshape(5, 10);
	*/
	template <typename TP>
	void ArrayGPU<TP>::reshape(const int newRows, const int newCols)
	{
		if (newRows * newCols == this->_rows * this->_cols)
		{
			this->_rows = newRows;
			this->_cols = newCols;
		}
		else if(newRows == -1 && (this->_rows * this->_cols) % newCols == 0){
			this->_rows = (this->_rows * this->_cols) / newCols;
			this->_cols = newCols;
		}
		else if(newCols == -1 && (this->_rows * this->_cols) % newRows == 0){
			this->_cols = (this->_rows * this->_cols) / newRows;
			this->_rows = newRows;
		}
		else
			std::cerr << "\nError! New size and old size are not equal.";
	}

	/*
		Returns a copy of array as cpu pointer.
		Ex: float *a = A.cpu();
	*/
	template <typename TP>
	TP* ArrayGPU<TP>::cpu() const{
		TP *array_h = (TP *)malloc(this->_rows * this->_cols * sizeof(TP));

		CUDA_CALL(cudaMemcpy(array_h, this->mat, this->_rows * this->_cols * sizeof(TP), cudaMemcpyDeviceToHost));
		return array_h;
	}

	// ####################### ARRAY ELEMENT GETTER SETTER FUNCTIONS ############################

	/*
		Returns element at idx as array object.
		Arguments:
		* idx - idx of element
		Ex: auto B = A.at(0);
	*/
	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::at(const int idx) const
	{
		ArrayGPU<TP> res(1, 1);
		res.mat = (this->mat + idx);
		return res;
	}

	/*
		Returns element at (r, c) as array object.
		Arguments:
		* r - r to access from
		* c - c to access from
		Ex: auto B = A.at(5, 2);
	*/
	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::at(const int r, const int c) const
	{
		return this->at(r * this->_cols + c);
	}

	// ####################### DOT PRODUCT ############################
	/*
		Returns dot product of two arrays
		Arguments:
		* B - second array
		Ex: auto C = A.dot(B);
	*/
	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::dot(const ArrayGPU<TP> &B) const
	{
		// condition for dot product
		if (this->_cols == B._rows)
		{
			ArrayGPU<TP> res(this->_rows, B._cols);

			const float alpha = 1.0f;
			const float beta = 0.0f;

			// C = A . B k lie.
			cublasSgemm(cbls_handle, //
						CUBLAS_OP_N, CUBLAS_OP_N,
						B._cols, this->_rows, this->_cols, // B _cols, A _rows, A _cols
						&alpha,
						B.mat, B._cols,		   // B, B _cols
						this->mat, this->_cols, // A, A _cols
						&beta,
						res.mat, B._cols); // C, B _cols

			return res;
		}
		else
		{
			std::cerr << "\nError in dot! Check arguments";
			return np::ArrayGPU<TP>(1, 1, 0);
		}
	}
	/*
		Returns dot product of two arrays. First is transposed
		Arguments:
		* B - second array
		Ex: auto C = A.Tdot(B);
		Is same as - auto C = A.T().dot(B)
	*/
	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::Tdot(const ArrayGPU<TP> &B) const
	{
		if (this->_rows == B._rows)
		{
			ArrayGPU<TP> res(this->_cols, B._cols);

			const float alpha = 1.0f;
			const float beta = 0.0f;

			// C = AT . B
			cublasSgemm(cbls_handle, //
						CUBLAS_OP_N, CUBLAS_OP_T,
						B._cols, this->_cols, this->_rows, // B _cols, A _cols, A _rows
						&alpha,
						B.mat, B._cols,		   // B, B _cols
						this->mat, this->_cols, // A, A _cols
						&beta,
						res.mat, B._cols); // C, B _cols

			return res;
		}
		else
		{
			std::cerr << "\nError in Tdot! Check arguments";
			return np::ArrayGPU<TP>(1, 1, 0);
		}
	}
	/*
		Returns dot product of two arrays. Second is transposed
		Arguments:
		* B - second array
		Ex: auto C = A.dotT(B);
		Is same as - auto C = A.dot(B.T())
	*/
	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::dotT(const ArrayGPU<TP> &B) const
	{
		if (this->_cols == B._cols)
		{
			ArrayGPU<TP> res(this->_rows, B._rows);

			const float alpha = 1.0f;
			const float beta = 0.0f;

			cublasSgemm(cbls_handle, //
						CUBLAS_OP_T, CUBLAS_OP_N,
						B._rows, this->_rows, this->_cols, // B _cols, A _rows, A _cols
						&alpha,
						B.mat, B._cols,		   // B, B _cols
						this->mat, this->_cols, // A, A _cols
						&beta,
						res.mat, B._rows); // C, B _cols

			return res;
		}
		else
		{
			std::cerr << "\nError in dotT ! Check arguments";
			return np::ArrayGPU<TP>(1, 1, 0);
		}
	}

	template <typename TP>
	template <char OP>
	ArrayGPU<TP> ArrayGPU<TP>::applyOp(const ArrayGPU<TP> &B) const
	{
		if (this->_rows == 1 && this->_cols == 1)
		{
			// A is scalar
			ArrayGPU<TP> res(B._rows, B._cols);

			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(std::min<int>(GPU_NUM_SM*2, ceil(res.size(), block.x)));

			kernelScalarOpMat<TP, OP><<<grid, block>>>(this->mat, B.mat, res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}
		else if (B._rows == 1 && B._cols == 1)
		{
			// B is scalar
			ArrayGPU<TP> res(this->_rows, this->_cols);

			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(std::min<int>(GPU_NUM_SM*2, ceil(res.size(), block.x)));

			kernelMatOpScalar<TP, OP><<<grid, block>>>(this->mat, B.mat, res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}
		// if A is vector
		// A vector ki dim, is equal to either col or row of B
		// row vector. will extend along _cols if possible. (prioritising in case of square matrix)
		// vice versa for _cols

		else if ((this->_cols == 1 && this->_rows == B._rows) || (this->_rows == 1 && this->_cols == B._rows))
		{
			// along _rows add kr
			ArrayGPU<TP> res(B._rows, B._cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(std::min<int>(GPU_NUM_SM*2, ceil(res.size(), block.x)));
			kernelVecOpMatAlongCols<TP, OP><<<grid, block>>>(this->mat, B.mat, res.mat, res.size(), B._cols);
			cudaDeviceSynchronize();

			return res;
		}
		else if ((this->_cols == 1 && this->_rows == B._cols) || (this->_rows == 1 && this->_cols == B._cols))
		{
			// along _cols add kr
			ArrayGPU<TP> res(B._rows, B._cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(std::min<int>(GPU_NUM_SM*2, ceil(res.size(), block.x)));
			kernelVecOpMatAlongRows<TP, OP><<<grid, block>>>(this->mat, B.mat, res.mat, res.size(), B._cols);
			cudaDeviceSynchronize();

			return res;
		}
		// B is vetor
		// B vector ki dim, is eq to either col or row of B
		// row vector. will extend along _cols if possible. (prioritising in case of square matrix)
		else if ((B._cols == 1 && this->_rows == B._rows) || (B._rows == 1 && this->_rows == B._cols))
		{
			// along _rows add kr
			ArrayGPU<TP> res(this->_rows, this->_cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(std::min<int>(GPU_NUM_SM*2, ceil(res.size(), block.x)));
			kernelMatOpVecAlongCols<TP, OP><<<grid, block>>>(this->mat, B.mat, res.mat, res.size(), this->_cols);
			cudaDeviceSynchronize();

			return res;
		}
		else if ((B._cols == 1 && this->_cols == B._rows) || (B._rows == 1 && this->_cols == B._cols))
		{
			// along _cols add kr
			ArrayGPU<TP> res(this->_rows, this->_cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(std::min<int>(GPU_NUM_SM*2, ceil(res.size(), block.x)));
			kernelMatOpVecAlongRows<TP, OP><<<grid, block>>>(this->mat, B.mat, res.mat, res.size(), this->_cols);
			cudaDeviceSynchronize();

			return res;
		}
		else if (this->_rows == B._rows && this->_cols == B._cols)
		{
			// A and B both are matrices of same dimensions
			ArrayGPU<TP> res(this->_rows, this->_cols);
			const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
			dim3 block(BLOCK_SIZE);
			dim3 grid(std::min<int>(GPU_NUM_SM*2, ceil(res.size(), block.x)));
			kernelMatOpMat<TP, OP><<<grid, block>>>(this->mat, B.mat, res.mat, res.size());
			cudaDeviceSynchronize();
			return res;
		}
		else
		{
			std::cerr << "\nError in applyOP! Check arguments";
			return np::ArrayGPU<TP>(1, 1, 0);
		}
	}

	template <typename TP>
	template <char OP>
	ArrayGPU<TP> ArrayGPU<TP>::applyOp(const TP Scalar) const
	{
		ArrayGPU<TP> res(this->_rows, this->_cols);
		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(std::min<int>(GPU_NUM_SM*2, ceil(res.size(), block.x)));
		kernelMatOpScalar<TP, OP><<<grid, block>>>(this->mat, Scalar, res.mat, res.size());
		cudaDeviceSynchronize();
		return res;
	}

	// add functions
	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::operator+(const ArrayGPU<TP> &B) const
	{
		return this->applyOp<NP_OP_ADD>(B);
	}

	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::operator+(const TP Scalar) const
	{
		return this->applyOp<NP_OP_ADD>(Scalar);
	}

	template <typename TP>
	ArrayGPU<TP> operator+(const TP Scal, const ArrayGPU<TP> &B)
	{
		// A is scalar
		ArrayGPU<TP> res(B._rows, B._cols);

		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(std::min<int>(GPU_NUM_SM*2, ceil(res.size(), block.x)));

		kernelScalarOpMat<TP, NP_OP_ADD><<<grid, block>>>(Scal, B.mat, res.mat, res.size());
		cudaDeviceSynchronize();
		return res;
	}

	// subtraction
	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::operator-(const ArrayGPU<TP> &B) const
	{
		return this->applyOp<NP_OP_SUB>(B);
	}

	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::operator-(const TP Scalar) const
	{
		return this->applyOp<NP_OP_SUB>(Scalar);
	}

	template <typename TP>
	ArrayGPU<TP> operator-(const TP Scal, const ArrayGPU<TP> &B)
	{
		// A is scalar
		ArrayGPU<TP> res(B._rows, B._cols);

		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(std::min<int>(GPU_NUM_SM*2, ceil(res.size(), block.x)));

		kernelScalarOpMat<TP, NP_OP_SUB><<<grid, block>>>(Scal, B.mat, res.mat, res.size());
		cudaDeviceSynchronize();
		return res;
	}

	// multiplication
	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::operator*(const ArrayGPU<TP> &B) const
	{
		return this->applyOp<NP_OP_MUL>(B);
	}

	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::operator*(const TP Scalar) const
	{
		return this->applyOp<NP_OP_MUL>(Scalar);
	}

	template <typename TP>
	ArrayGPU<TP> operator*(const TP Scal, const ArrayGPU<TP> &B)
	{
		// A is scalar
		ArrayGPU<TP> res(B._rows, B._cols);

		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(std::min<int>(GPU_NUM_SM*2, ceil(res.size(), block.x)));

		kernelScalarOpMat<TP, NP_OP_MUL><<<grid, block>>>(Scal, B.mat, res.mat, res.size());
		cudaDeviceSynchronize();
		return res;
	}

	// division
	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::operator/(const ArrayGPU<TP> &B) const
	{
		return this->applyOp<NP_OP_DIV>(B);
	}

	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::operator/(const TP Scalar) const
	{
		return this->applyOp<NP_OP_DIV>(Scalar);
	}

	template <typename TP>
	ArrayGPU<TP> operator/(const TP Scal, const ArrayGPU<TP> &B)
	{
		// A is scalar
		ArrayGPU<TP> res(B._rows, B._cols);

		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(std::min<int>(GPU_NUM_SM*2, ceil(res.size(), block.x)));

		kernelScalarOpMat<TP, NP_OP_DIV><<<grid, block>>>(Scal, B.mat, res.mat, res.size());
		cudaDeviceSynchronize();
		return res;
	}

	// unary negation operator
	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::operator-() const
	{
		ArrayGPU<TP> res(this->_rows, this->_cols);
		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(std::min<int>(GPU_NUM_SM*2, ceil(res.size(), block.x)));
		kernelMatOpScalar<TP, NP_OP_MUL><<<grid, block>>>(this->mat, -1, res.mat, res.size());
		cudaDeviceSynchronize();
		return res;
	}

	// returns an array of 0s and 1s depending on true or false of the conditions.
	//  element wise comparison

	// <
	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::operator<(const ArrayGPU<TP> &B) const
	{
		return this->applyOp<NP_OP_LESS_THAN>(B);
	}

	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::operator<(const TP Scalar) const
	{
		return this->applyOp<NP_OP_LESS_THAN>(Scalar);
	}

	template <typename TP>
	ArrayGPU<TP> operator<(const TP Scal, const ArrayGPU<TP> &B)
	{
		// A is scalar
		ArrayGPU<TP> res(B._rows, B._cols);

		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(std::min<int>(GPU_NUM_SM*2, ceil(res.size(), block.x)));

		kernelScalarOpMat<TP, NP_OP_LESS_THAN><<<grid, block>>>(Scal, B.mat, res.mat, res.size());
		cudaDeviceSynchronize();
		return res;
	}

	// <=
	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::operator<=(const ArrayGPU<TP> &B) const
	{
		return this->applyOp<NP_OP_LESS_THAN_EQ>(B);
	}

	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::operator<=(const TP Scalar) const
	{
		return this->applyOp<NP_OP_LESS_THAN_EQ>(B);
	}

	template <typename TP>
	ArrayGPU<TP> operator<=(const TP Scal, const ArrayGPU<TP> &B)
	{
		// A is scalar
		ArrayGPU<TP> res(B._rows, B._cols);

		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(std::min<int>(GPU_NUM_SM*2, ceil(res.size(), block.x)));

		kernelScalarOpMat<TP, NP_OP_LESS_THAN_EQ><<<grid, block>>>(Scal, B.mat, res.mat, res.size());
		cudaDeviceSynchronize();
		return res;
	}

	// >
	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::operator>(const ArrayGPU<TP> &B) const
	{
		return this->applyOp<NP_OP_GREATER_THAN>(B);
	}

	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::operator>(const TP Scalar) const
	{
		return this->applyOp<NP_OP_GREATER_THAN>(Scalar);
	}

	template <typename TP>
	ArrayGPU<TP> operator>(const TP Scal, const ArrayGPU<TP> &B)
	{
		// A is scalar
		ArrayGPU<TP> res(B._rows, B._cols);

		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(std::min<int>(GPU_NUM_SM*2, ceil(res.size(), block.x)));

		kernelScalarOpMat<TP, NP_OP_GREATER_THAN><<<grid, block>>>(Scal, B.mat, res.mat, res.size());
		cudaDeviceSynchronize();
		return res;
	}

	// >=
	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::operator>=(const ArrayGPU<TP> &B) const
	{
		return this->applyOp<NP_OP_GREATER_THAN_EQ>(B);
	}

	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::operator>=(const TP Scalar) const
	{
		return this->applyOp<NP_OP_GREATER_THAN_EQ>(B);
	}

	template <typename TP>
	ArrayGPU<TP> operator>=(const TP Scal, const ArrayGPU<TP> &B)
	{
		// A is scalar
		ArrayGPU<TP> res(B._rows, B._cols);

		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(std::min<int>(GPU_NUM_SM*2, ceil(res.size(), block.x)));

		kernelScalarOpMat<TP, NP_OP_GREATER_THAN_EQ><<<grid, block>>>(Scal, B.mat, res.mat, res.size());
		cudaDeviceSynchronize();
		return res;
	}

	// ==
	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::operator==(const ArrayGPU<TP> &B) const
	{
		return this->applyOp<NP_OP_EQ>(B);
	}

	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::operator==(const TP Scalar) const
	{
		return this->applyOp<NP_OP_EQ>(B);
	}

	template <typename TP>
	ArrayGPU<TP> operator==(const TP Scal, const ArrayGPU<TP> &B)
	{
		// A is scalar
		ArrayGPU<TP> res(B._rows, B._cols);

		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(std::min<int>(GPU_NUM_SM*2, ceil(res.size(), block.x)));

		kernelScalarOpMat<TP, NP_OP_EQ><<<grid, block>>>(Scal, B.mat, res.mat, res.size());
		cudaDeviceSynchronize();
		return res;
	}

	// !=
	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::operator!=(const ArrayGPU<TP> &B) const
	{
		return this->applyOp<NP_OP_NOT_EQ>(B);
	}

	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::operator!=(const TP Scalar) const
	{
		return this->applyOp<NP_OP_NOT_EQ>(B);
	}

	template <typename TP>
	ArrayGPU<TP> operator!=(const TP Scal, const ArrayGPU<TP> &B)
	{
		// A is scalar
		ArrayGPU<TP> res(B._rows, B._cols);

		const int BLOCK_SIZE = GPU_NUM_CUDA_CORE;
		dim3 block(BLOCK_SIZE);
		dim3 grid(std::min<int>(GPU_NUM_SM*2, ceil(res.size(), block.x)));

		kernelScalarOpMat<TP, NP_OP_NOT_EQ><<<grid, block>>>(Scal, B.mat, res.mat, res.size());
		cudaDeviceSynchronize();
		return res;
	}

	template <typename TP>
	template <char F>
	ArrayGPU<TP> ArrayGPU<TP>::applyReductionF(const int axis) const
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
				kernelReduceF<TP, 64 * 2, F><<<grid, block>>>(this->mat, tmp_d, this->size());
				cudaDeviceSynchronize();
				// please guarantee that BLOCK_SIZE > grid.x. otherwise multiple kernel calls will have to be made.
				kernelReduceF<TP, 64 * 2, F><<<1, block>>>(tmp_d, res.mat, grid.x);
				cudaDeviceSynchronize();
				break;
			default:
				kernelReduceF<TP, 128 * 2, F><<<grid, block>>>(this->mat, tmp_d, this->size());
				cudaDeviceSynchronize();
				// please guarantee that BLOCK_SIZE > grid.x. otherwise multiple kernel calls will have to be made.
				kernelReduceF<TP, 128 * 2, F><<<1, block>>>(tmp_d, res.mat, grid.x);
				cudaDeviceSynchronize();
				break;
			}
			CUDA_CALL(cudaFree(tmp_d));
			return res;
		}
		else if (axis == 0)
		{
			// sum along columns. dimension=numCols
			auto ans = (this->T()).applyReductionF<F>(1);
			ans.reshape(1, -1);
			return ans;
		}
		else if (axis == 1)
		{
			// sum along _rows. output dim = numRows
			ArrayGPU<TP> res(this->_rows, 1);

			const int BLOCK_SIZE = ((GPU_NUM_CUDA_CORE == 64) ? 64 : 128) * 2;
			dim3 block(BLOCK_SIZE);
			dim3 grid(std::min<int>(ceil(this->_cols, block.x), GPU_NUM_SM * 2));

			TP *tmp_d;
			CUDA_CALL(cudaMalloc((void **)&tmp_d, sizeof(TP) * this->_rows * grid.x));

			switch (GPU_NUM_CUDA_CORE)
			{
			case 64:
				for (int i = 0; i < this->_rows; ++i)
				{
					kernelReduceF<TP, 64 * 2, F><<<grid, block>>>(this->mat + i * this->_cols, tmp_d + i * grid.x, this->_cols);
				}

				cudaDeviceSynchronize();

				for (int i = 0; i < this->_rows; ++i)
				{
					kernelReduceF<TP, 64 * 2, F><<<1, block>>>(tmp_d + i * grid.x, res.mat + i, grid.x);
				}

				cudaDeviceSynchronize();
				break;
			default:
				for (int i = 0; i < this->_rows; ++i)
				{
					kernelReduceF<TP, 128 * 2, F><<<grid, block>>>(this->mat + i * this->_cols, tmp_d + i * grid.x, this->_cols);
				}

				cudaDeviceSynchronize();

				for (int i = 0; i < this->_rows; ++i)
				{
					kernelReduceF<TP, 128 * 2, F><<<1, block>>>(tmp_d + i * grid.x, res.mat + i, grid.x);
				}

				cudaDeviceSynchronize();
			}
			return res;
		}
		else
		{
			std::cerr << "\nError in applyReductionF! Check arguments";
			return np::ArrayGPU<TP>(1, 1, 0);
		}
	}

	// sum. along axis or total
	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::sum(const int axis) const
	{
		return this->applyReductionF<NP_REDUCE_SUM>(axis);
	}

	// min. along axis or total
	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::min(const int axis) const
	{
		return this->applyReductionF<NP_REDUCE_MIN>(axis);
	}

	// max. along axis or total
	template <typename TP>
	ArrayGPU<TP> ArrayGPU<TP>::max(const int axis) const
	{
		return this->applyReductionF<NP_REDUCE_MAX>(axis);
	}

	template <typename TP>
	template <char F>
	ArrayGPU<int> ArrayGPU<TP>::applyReductionArgF(const int axis) const
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
			CUDA_CALL(cudaMalloc((void **)&tmp_A_Idx_d, sizeof(int) * grid.x));

			switch (GPU_NUM_CUDA_CORE)
			{
			case 64:
				kernelReduceArgF<TP, 64 * 2, F><<<grid, block>>>(this->mat, tmp_A_d, tmp_A_Idx_d, this->size());
				cudaDeviceSynchronize();
				// please guarantee that BLOCK_SIZE > grid.x. otherwise multiple kernel calls will have to be made.
				if(grid.x == 1){
					resIdx.mat = tmp_A_Idx_d;
					return resIdx;
				}
				kernelReduceArgF<TP, 64 * 2, F><<<1, block>>>(tmp_A_d, tmp_A_Idx_d, res.mat, resIdx.mat, grid.x);
				cudaDeviceSynchronize();
				break;
			default:
				kernelReduceArgF<TP, 128 * 2, F><<<grid, block>>>(this->mat, tmp_A_d, tmp_A_Idx_d, this->size());
				cudaDeviceSynchronize();
				// please guarantee that BLOCK_SIZE > grid.x. otherwise multiple kernel calls will have to be made.
				if(grid.x == 1){
					resIdx.mat = tmp_A_Idx_d;
					return resIdx;
				}
				kernelReduceArgF<TP, 128 * 2, F><<<1, block>>>(tmp_A_d, tmp_A_Idx_d, res.mat, resIdx.mat, grid.x);
				cudaDeviceSynchronize();
			}

			CUDA_CALL(cudaFree(tmp_A_d));

			return resIdx;
		}
		else if (axis == 0)
		{
			// sum along columns. dimension=numCols
			auto ans = this->T().applyReductionArgF<F>(1);
			ans.reshape(1, -1);
			return ans;
		}
		else if (axis == 1)
		{
			// sum along _rows. output dim = numRows
			ArrayGPU<TP> res(this->_rows, 1);
			ArrayGPU<int> resIdx(this->_rows, 1);

			const int BLOCK_SIZE = ((GPU_NUM_CUDA_CORE == 128) ? 128 : 64) * 2;
			dim3 block(BLOCK_SIZE);
			dim3 grid(std::min<int>(GPU_NUM_SM * 2, ceil(this->_cols, block.x)));
			std::cout<<"\nBLOCK: "<<block.x<<" GRID: "<<grid.x<<std::endl;

			TP *tmp_A_d;
			CUDA_CALL(cudaMalloc((void **)&tmp_A_d, sizeof(TP) * this->_rows * grid.x));
			int *tmp_A_Idx_d;
			CUDA_CALL(cudaMalloc((void **)&tmp_A_Idx_d, sizeof(int) * this->_rows * grid.x));

			switch (GPU_NUM_CUDA_CORE)
			{
			case 64:
				for (int i = 0; i < this->_rows; ++i)
					kernelReduceArgF<TP, 64 * 2, F><<<grid, block>>>(this->mat + i * this->_cols, tmp_A_d + i * grid.x, tmp_A_Idx_d + i * grid.x, this->_cols);
				cudaDeviceSynchronize();

				if(grid.x == 1){
					resIdx.mat = tmp_A_Idx_d;
					return resIdx;
				}
				for (int i = 0; i < this->_rows; ++i)
					kernelReduceArgF<TP, 64 * 2, F><<<1, block>>>(tmp_A_d + i * grid.x, tmp_A_Idx_d + i * grid.x, res.mat + i, resIdx.mat + i, grid.x);
				cudaDeviceSynchronize();
				break;
			default:
				for (int i = 0; i < this->_rows; ++i)
					kernelReduceArgF<TP, 128 * 2, F><<<grid, block>>>(this->mat + i * this->_cols, tmp_A_d + i * grid.x, tmp_A_Idx_d + i * grid.x, this->_cols);

				cudaDeviceSynchronize();
				if(grid.x == 1){
					resIdx.mat = tmp_A_Idx_d;
					return resIdx;
				}
				for (int i = 0; i < this->_rows; ++i)
					kernelReduceArgF<TP, 128 * 2, F><<<1, block>>>(tmp_A_d + i * grid.x, tmp_A_Idx_d + i * grid.x, res.mat + i, resIdx.mat + i, grid.x);

				cudaDeviceSynchronize();

			}

			return resIdx;
		}
		else
		{
			std::cerr << "\nError in applyReductionArgF! Check arguments";
			return np::ArrayGPU<int>(1, 1, 0);
		}
	}

	// argmin
	// min along axis or total
	template <typename TP>
	ArrayGPU<int> ArrayGPU<TP>::argmin(const int axis) const
	{
		return this->applyReductionArgF<NP_REDUCE_ARGMIN>(axis);
	}

	// argmax
	template <typename TP>
	ArrayGPU<int> ArrayGPU<TP>::argmax(const int axis) const
	{
		return this->applyReductionArgF<NP_REDUCE_ARGMAX>(axis);
	}

	template <typename TP>
	ArrayGPU<TP>::~ArrayGPU()
	{
		--(*this->ref_count);
		if(*this->ref_count == 0){
			cudaFree(this->mat);
			free(ref_count);
		}
	}
}

#endif