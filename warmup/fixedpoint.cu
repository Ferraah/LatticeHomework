

#include <iostream>
#include <vector>
#include <limits.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "random_generator.hpp"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void fp_min(int *d_arr, int *d_min, int N){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx < N){
        if(d_arr[idx] < *d_min){
            *d_min = d_arr[idx];
        }
    }


}



int main(){

    // Setting up the environment 
    int N = 1 << 21; // 2^20 elements
    int block_size = 1024;
    int grid_size = (N + block_size - 1) /  block_size;

    int min = INT_MAX; // Setting the minimum value to the maximum value of an integer
    int *arr = new int[N];
    int *d_min, *d_arr;

    // Allocate memory on the device
    cudaMalloc(&d_min, sizeof(int));
    cudaMalloc(&d_arr, N*sizeof(int));

    int seed = 10;
    RandomGenerator<int> rg(seed);

    // Setting array values
    for(int i = 0; i < N; i++){
        arr[i] = rg.getRandomNumber(1,N);
        //arr[i] = i; 
    }
    arr[2000] = -1;

    // Copying data to the device
    cudaMemcpy(d_min, &min, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr, arr, N*sizeof(int), cudaMemcpyHostToDevice);

    std::cout << "Starting min: " << min << std::endl;
    std::cout << "Grid Size: " << grid_size << " - Block Size: " <<  block_size << std::endl;

    int old_min = min + 1;
    int MAX_DATA_RACES = N-1;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // MAX_DATA_RACES+2 iteration at its worst
    for(int i=0; i<MAX_DATA_RACES+2; i++){

        if(min == old_min){
            break;
        }

        old_min = min;

        fp_min<<<grid_size,  block_size>>>(d_arr, d_min, N);
        //gpuErrchk( cudaPeekAtLastError() );

        //cudaDeviceSynchronize();

        cudaMemcpy(&min, d_min, sizeof(int), cudaMemcpyDeviceToHost);
        std::cout << "I: "<< i << " - Min: " << min << std::endl;

    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_elapsed = 0;
    cudaEventElapsedTime(&time_elapsed, start, stop);

    std::cout << "Kernel execution time: " << time_elapsed << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}