#ifndef BOOL_MATRIX_GPU_H
#define BOOL_MATRIX_GPU_H

#include <iostream>
#include <cuda_runtime.h>
#include "gpuAssert.hpp"


struct BoolMatrixGPU{
    size_t rows;
    size_t size;
    size_t *cols = nullptr;
    size_t *offsets = nullptr;  // Pointer to the start of each row
    size_t *d_offsets = nullptr;  // Pointer to the start of each row
    bool* data = nullptr;  // Pointer to raw data

    // Constructor (uses externally allocated memory)
    BoolMatrixGPU(size_t rows, int *u)
        : rows(rows), size(0) {
            _(cudaMalloc(&cols, rows * sizeof(size_t)));
            
            size_t *host_cols = new size_t[rows];
            for(size_t i = 0; i < rows; i++){
                host_cols[i] = u[i]+1;
            }

            _(cudaMemcpy(cols, host_cols, rows * sizeof(size_t), cudaMemcpyHostToDevice));
            assert(cols != nullptr);

            // Create the offsets array
            offsets = new size_t[rows];
            offsets[0] = 0;
            for(size_t i = 1; i < rows; i++){
                offsets[i] = offsets[i-1] + host_cols[i-1];
            }
            
            _(cudaMalloc(&d_offsets, rows * sizeof(size_t)));
            _(cudaMemcpy(d_offsets, offsets, rows * sizeof(size_t), cudaMemcpyHostToDevice));

            // Matrix initialization
            size = offsets[rows-1] + host_cols[rows-1];
            _(cudaMalloc(&data, size * sizeof(bool)));
            _(cudaMemset(data, 1, size * sizeof(bool)));

            delete[] host_cols;
        }

    // Device copy constructor
    BoolMatrixGPU(const BoolMatrix& other)
        : rows(other.rows), size(other.size) { 
        
        cudaStream_t stream[3];
        for(int i = 0; i < 3; i++){
            _(cudaStreamCreate(&stream[i]));
        }

        assert(other.data != nullptr);
        assert(other.cols != nullptr);
        assert(other.offsets != nullptr);


        _(cudaMalloc(&data, size * sizeof(bool)));
        _(cudaMemcpyAsync(data, other.data, size * sizeof(bool), cudaMemcpyHostToDevice, stream[0]));
        _(cudaMalloc(&cols, rows * sizeof(size_t)));
        _(cudaMemcpyAsync(cols, other.cols, rows * sizeof(size_t), cudaMemcpyHostToDevice, stream[1]));
        _(cudaMalloc(&d_offsets, rows * sizeof(size_t)));
        _(cudaMemcpyAsync(d_offsets, other.offsets, rows * sizeof(size_t), cudaMemcpyHostToDevice, stream[2]));
        
        for(int i = 0; i < 3; i++){
            _(cudaStreamSynchronize(stream[i]));
            _(cudaStreamDestroy(stream[i]));
        }

    }

    // Deep copy constructor
    BoolMatrixGPU(const BoolMatrixGPU& other)
        : rows(other.rows), size(other.size)  {
        _(cudaMalloc(&data, size * sizeof(bool)));
        _(cudaMemcpy(data, other.data, size * sizeof(bool), cudaMemcpyDeviceToDevice));
        _(cudaMalloc(&cols, rows * sizeof(size_t)));
        _(cudaMemcpy(cols, other.cols, rows * sizeof(size_t), cudaMemcpyDeviceToDevice));
        _(cudaMalloc(&d_offsets, rows * sizeof(size_t)));
        _(cudaMemcpy(d_offsets, other.d_offsets, rows * sizeof(size_t), cudaMemcpyDeviceToDevice));

        offsets = new size_t[rows];
        memcpy(offsets, other.offsets, rows * sizeof(size_t));
    }

    // Destructor
    ~BoolMatrixGPU() {
        if(data != nullptr) _(cudaFree(data));
        if(cols != nullptr) _(cudaFree(cols));
        if(d_offsets != nullptr) _(cudaFree(d_offsets));
        if(offsets != nullptr) delete[] offsets;
    }
        // Overload operator[] for 2D indexing
    __host__ bool *operator[](size_t row)
    {
        assert(offsets != nullptr);
        return &data[offsets[row]]; // Return pointer to the start of the row
    }

    // Const version for read-only access
    __host__ const bool *operator[](size_t row) const
    {

        assert(offsets != nullptr);
        return &data[offsets[row]];
    }

};

#endif