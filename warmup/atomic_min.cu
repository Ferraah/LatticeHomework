#include <iostream>
#include <vector>
#include <limits.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "random_generator.hpp"

#define _(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void atomic_min_naive_kernel(int *d_arr, int *d_min, int N){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx < N){
        atomicMin(d_min, d_arr[idx]);
    }
    
}



int main(){

    // Setting up the environment 
    int N = 1 << 20; // N elements
    int block_size = 1024;
    int grid_size = (N + block_size - 1) /  block_size;

    int min = INT_MAX; // Setting the minimum value to the maximum value of an integer
    int *arr = new int[N];
    int *d_min, *d_arr;

    // Allocate memory on the device
    _(cudaMalloc(&d_min, sizeof(int)));
    _(cudaMalloc(&d_arr, N * sizeof(int)));

    int seed = 10;
    RandomGenerator<int> rg(seed);

    // Setting array values
    rg.populateArray(arr, N, 1, N);
    arr[100] = -1;

    // Copying data to the device
    _(cudaMemcpy(d_min, &min, sizeof(int), cudaMemcpyHostToDevice));
    _(cudaMemcpy(d_arr, arr, N * sizeof(int), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    _(cudaEventCreate(&start));
    _(cudaEventCreate(&stop));

    cudaEventRecord(start);

    // Kernel call
    atomic_min_naive_kernel<<<grid_size,  block_size>>>(d_arr, d_min, N);
    _( cudaPeekAtLastError() );

    cudaDeviceSynchronize();

    _(cudaMemcpy(&min, d_min, sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "Min value: " << min << std::endl;


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_elapsed = 0;
    cudaEventElapsedTime(&time_elapsed, start, stop);

    std::cout << "Kernel execution time: " << time_elapsed << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}