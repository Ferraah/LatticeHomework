
#include <iostream>
#include <algorithm>
#include <vector>
#include <limits.h>
#include <chrono>

#include "random_generator.hpp"


void seq_min(int vec[], int N, int &min){

    for(int i=0; i < N; i++){
        if(vec[i] < min){
            min = vec[i];
        }
    }

}

int main(){

    int N = 1 << 20;
     
    int min_value = INT_MAX;
    int arr[N];

    int seed = 1;
    RandomGenerator<int> random(seed);

    // Initialize the vector with random values
    for(int i = 0; i < N; i++){
        arr[i] = random.getRandomNumber(1, N);
    }

    // Set the minimym value, for example at index 100
    arr[100] = 0;


    auto start = std::chrono::high_resolution_clock::now();
    seq_min(arr, N, min_value);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken by seq_min: " << duration.count()*1000 << " ms" << std::endl;
    std::cout << N << std::endl;
    std::cout << "The minimum value in the vector is: " << min_value << std::endl;
    return 0;
}