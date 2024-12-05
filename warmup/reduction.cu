


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

__global__ void reduce_min_1_kernel(int *input, int *output, int N, int s){
    unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;

    if(tid < N && tid + s < N){

        if(input[tid] < input[tid + s]){
            output[tid] = input[tid];
        }else{
            output[tid] = input[tid + s];
        }
    }

}

__global__ void reduce_min_2_kernel(int *input, int *output, int N){
    extern __shared__ int sdata[];
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;

    // Copy all elements to shared memory of block
    sdata[tid] = input[i];
    __syncthreads();
    
    for(unsigned int s = blockDim.x/2; s > 0; s >>= 1){
        if(tid < s){
            if(sdata[tid + s] < sdata[tid]){
                sdata[tid] = sdata[tid + s];
            }
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if(tid == 0){
        output[blockIdx.x] = sdata[0];
    }


}

void reduce_min_1(int *input, int *output, int N, int *min){

    int block_size = 1024;
    int grid_size = (N + block_size - 1) /  block_size;
    
    for (int s = 1; s < N; s *= 2){
        reduce_min_1_kernel<<<grid_size,  block_size>>>(input, output, N, s);
        _( cudaPeekAtLastError() );
        _(cudaDeviceSynchronize());
    }
    
    _(cudaMemcpy(min, output, sizeof(int), cudaMemcpyDeviceToHost));

}

void reduce_min_2(int *input, int N, int *min){

    int block_size = 1024;
    int grid_size = (N + block_size - 1) /  block_size;
  
    int *output = new int[grid_size];
    int *d_output;
    _(cudaMalloc(&d_output, grid_size*sizeof(int))); 

    reduce_min_2_kernel<<<grid_size,  block_size, block_size*sizeof(int)>>>(input, d_output, N);
    _( cudaPeekAtLastError() );

    _(cudaMemcpy( output, d_output, grid_size*sizeof(int), cudaMemcpyDeviceToHost));

    // Serially reduce the output array
    for(int i = 0; i < grid_size; i++){
        if(output[i] < *min){
            *min = output[i];
        }
    }

}




int main(){

    // Setting up the environment 
    int N = 1 << 20; // 2^20 elements

    int min; // Setting the minimum value to the maximum value of an integer
    int *arr = new int[N];
    int *d_arr;

    // Allocate memory on the device
    _(cudaMalloc(&d_arr, N*sizeof(int)));

    int seed = 10;
    RandomGenerator<int> rg(seed);

    cudaEvent_t start, stop;
    _(cudaEventCreate(&start));
    _(cudaEventCreate(&stop));

    float time_elapsed = 0;
    // -----------------------------------------------------------------------

    // Version 1 

    // Setting array values
    rg.populateArray(arr, N, 1, N);
    arr[100] = -1;
    _(cudaMemcpy(d_arr, arr, N*sizeof(int), cudaMemcpyHostToDevice));

    _(cudaEventRecord(start));

    reduce_min_1(d_arr, d_arr, N, &min);
    std::cout << "Final min: " << min << std::endl;

    _(cudaEventRecord(stop));
    _(cudaEventSynchronize(stop));

    _(cudaEventElapsedTime(&time_elapsed, start, stop));
    std::cout << "Kernel execution time: " << time_elapsed << " ms" << std::endl;

    // Reset min and time_elapsed for Version 2
    min = INT_MAX;
    time_elapsed = 0.0f;

    // Version 2

    _(cudaMemcpy(d_arr, arr, N*sizeof(int), cudaMemcpyHostToDevice));

    _(cudaEventRecord(start));

    reduce_min_2(d_arr, N, &min);
    std::cout << "Final min: " << min << std::endl;

    _(cudaEventRecord(stop));
    _(cudaEventSynchronize(stop));

    _(cudaEventElapsedTime(&time_elapsed, start, stop));
    std::cout << "Kernel execution time: " << time_elapsed << " ms" << std::endl;


    // -----------------------------------------------------------------------
    // Free memory
    _(cudaFree( d_arr ));
    _(cudaEventDestroy(start));
    _(cudaEventDestroy(stop));

    return 0;
}