#include <iostream>
#include <cuda_runtime.h>

struct BoolMatrix
{
    size_t rows;
    size_t *cols = nullptr;
    size_t *offsets = nullptr; // Pointer to the start of each row
    bool *data = nullptr; // Pointer to raw data


    // Constructor (uses externally allocated memory)
    BoolMatrix(size_t rows, int *cols_int)
        : rows(rows){

            cols = new size_t[rows];
            for(size_t i = 0; i < rows; i++){
                cols[i] = cols_int[i];
            }

            assert(cols != nullptr);
            
            // Create the offsets array
            offsets = new size_t[rows];
            offsets[0] = 0;
            for(size_t i = 1; i < rows; i++){
                offsets[i] = offsets[i-1] + cols[i-1];
            }

            // Matrix initialization
            size_t size = offsets[rows-1] + cols[rows-1];
            data = new bool[size]; 
            memset(data, 1, size);
        }

    // Deep copy constructor
    BoolMatrix(const BoolMatrix &other)
        : rows(other.rows)
    {
        assert(other.cols != nullptr);
        assert(other.offsets != nullptr);
        assert(other.data != nullptr);

        size_t size = other.offsets[rows-1] + other.cols[rows-1];

        data = new bool[size];
        assert(other.data != nullptr);

        for (size_t i = 0; i < size; ++i)
        {
            data[i] = other.data[i];
        }

        // Copy the offsets and cols lengths
        offsets = new size_t[rows];
        cols = new size_t[rows];
        for(size_t i = 0; i < rows; i++){
            offsets[i] = other.offsets[i];
            cols[i] = other.cols[i];
        }

    }

    // Move constructor
    BoolMatrix(BoolMatrix &&other) noexcept
        : rows(other.rows), cols(other.cols), data(other.data), offsets(other.offsets)
    {
        other.data = nullptr;
        other.offsets = nullptr;
        other.cols = nullptr;
    }

    // Move assignment operator
    BoolMatrix &operator=(BoolMatrix &&other) noexcept
    {
        if (this != &other)
        {
            delete[] data;
            delete[] offsets;
            delete[] cols;
            rows = other.rows;
            cols = other.cols;
            data = other.data;
            offsets = other.offsets;
            other.data = nullptr;
            other.offsets = nullptr;
            other.cols = nullptr;
        }
        return *this;
    }

    // Function to find the index of the unique true value in a row
    size_t find_unique_true_index(size_t row) const
    {
        int trueIndex = -1;
        bool foundTrue = false;
        for (size_t col = 0; col < cols[row]; ++col)
        {
            if ((*this)[row][col])
            {
                if (foundTrue)
                {
                    return -1; // More than one true value found
                }
                trueIndex = col;
                foundTrue = true;
            }
        }
        return trueIndex;
    }

    // Overload operator[] for 2D indexing
    bool *operator[](size_t row)
    {
        size_t offset = 0;
        for(size_t i = 0; i < row; i++){
            offset += cols[i];
        }
        return &data[offset]; // Return pointer to the start of the row
    }

    // Const version for read-only access
    const bool *operator[](size_t row) const
    {

        size_t offset = 0;
        for(size_t i = 0; i < row; i++){
            offset += cols[i];
        }
        return &data[offset];
    }

    // Destructor
    ~BoolMatrix()
    {
        if (data)
        {
            delete[] data;
            delete[] offsets;
            delete[] cols;

            data = nullptr;
            offsets = nullptr;
            cols = nullptr;
        }
    }
};
