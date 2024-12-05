# PCCP homework
For compiling on Iris: 
```
module load toolchain/fosscuda
mkdir build && cd build
nvcc ../comb_optimization/comb_optimization_gpu.cu -o gpu -O3
g++  ../comb_optimization/comb_optimization.cpp -o cpu -O3
```
