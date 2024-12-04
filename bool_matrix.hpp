#include <iostream>
#include <cuda_runtime.h>

struct BoolMatrix {
    size_t rows;
    size_t cols;
    bool* data;  // Pointer to raw data

    // Constructor (uses externally allocated memory)
    __host__ __device__ BoolMatrix(size_t rows, size_t cols, bool* data)
        : rows(rows), cols(cols), data(data) {}

    // Copy constructor
    __host__ BoolMatrix(const BoolMatrix& other)
        : rows(other.rows), cols(other.cols) {
        size_t size = rows * cols;
        data = new bool[size];
        for (size_t i = 0; i < size; ++i) {
            data[i] = other.data[i];
        }
    }

    // Function to find the index of the unique true value in a row
    __host__ __device__ size_t find_unique_true_index(size_t row) const {
        int trueIndex = -1;
        bool foundTrue = false;
        for (size_t col = 0; col < cols; ++col) {
            if ((*this)[row][col]) {
                if (foundTrue) {
                    return -1;  // More than one true value found
                }
                trueIndex = col;
                foundTrue = true;
            }
        }
        return trueIndex;
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

