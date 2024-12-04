
#include <chrono>
#include <cstring>
#include <iostream>
#include <stack>
#include <utility>
#include <vector>
#include <numeric>

struct BitMask
{
    char *data;
    size_t m_rows;
    size_t n_cols;

    BitMask(size_t m_rows, size_t n_cols, int value)
    {
        this->m_rows = m_rows;
        this->n_cols = n_cols;
        data = new char[m_rows * n_cols];
        memset(data, value, m_rows * n_cols);
    }

    ~BitMask()
    {
        delete[] data;
    }

    bool operator[](size_t i) const
    {
        return data[i / 8] & (1 << (i % 8));
    }

    void set(size_t i)
    {
        data[i / 8] |= (1 << (i % 8));
    }

    void unset(size_t i)
    {
        data[i / 8] &= ~(1 << (i % 8));
    }

    void flip(size_t i)
    {
        data[i / 8] ^= (1 << (i % 8));
    }

    void clear(size_t size)
    {
        memset(data, 0, size);
    }

    size_t count_zeros(size_t from, size_t to) const
    {
        size_t count = 0;
        for (size_t i = from; i < to; i++)
        {
            if (!this->operator[](i))
                count++;
        }
        return count;
    }

    void copy_to(BitMask *other, size_t size)
    {
        memcpy(other->data, data, size);
    }

    // get first element set to zero, check if there is only one, otherwise throw an error
    size_t get_first_zero(size_t from, size_t to) const
    {
        size_t count = 0;
        size_t index = 0;
        for (size_t i = from; i < to; i++)
        {
            if (!this->operator[](i))
            {
                count++;
                index = i;
            }
        }
        if (count != 1)
            throw "Error: there should be only one element set to zero";
        return index;
    }
};