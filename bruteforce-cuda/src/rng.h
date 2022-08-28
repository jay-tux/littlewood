#ifndef _JAY_CUDA_RNG
#define _JAY_CUDA_RNG

#include <curand_kernel.h>
#include <ctime>
#include <cstdlib>
#include <cmath>

typedef curandState cuSt;

void setup_cpu();
void setup_gpu(cuSt *state, int blocks, int blocksize);
__global__ void setup_kernel(cuSt *state, ulong seed);

void generate_cpu(double *out);
__device__ void generate_gpu(cuSt *globalState, double *out);

__host__ __device__ void generate_any(cuSt *globalState, double *out);

#endif
