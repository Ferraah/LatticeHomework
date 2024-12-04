#include <iostream>
#include <cuda_runtime.h>

#define _(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

struct BoolMatrixGPU{
    size_t rows;
    size_t cols;
    bool* data = nullptr;  // Pointer to raw data

    // Constructor (uses externally allocated memory)
    BoolMatrixGPU(size_t rows, size_t cols, bool* data)
        : rows(rows), cols(cols), data(data) {}

    // Device copy constructor
    BoolMatrixGPU(const BoolMatrix& other)
        : rows(other.rows), cols(other.cols){

        assert(other.data != nullptr);
        size_t size = rows * cols;
        _(cudaMalloc(&data, size * sizeof(bool)));
        _(cudaMemcpy(data, other.data, size * sizeof(bool), cudaMemcpyHostToDevice));

    }

    // Deep copy constructor
    BoolMatrixGPU(const BoolMatrixGPU& other)
        : rows(other.rows), cols(other.cols) {
        size_t size = rows * cols;
        _(cudaMalloc(&data, size * sizeof(bool)));
        _(cudaMemcpy(data, other.data, size * sizeof(bool), cudaMemcpyDeviceToDevice));
    }

    // Destructor
    ~BoolMatrixGPU() {
        if(data != nullptr){
            _(cudaFree(data));
        }
    }

    // Overload operator[] for 2D indexing
    __host__ __device__ bool* operator[](size_t row) {
        return &data[row * cols];  // Return pointer to the start of the row
    }

    // Const version for read-only access
    __host__ __device__ const bool* operator[](size_t row) const {
        return &data[row * cols];
    }
};

