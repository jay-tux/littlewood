#include "rng.h"

void setup_cpu() { std::srand(std::time(nullptr)); }

void setup_gpu(cuSt *state, int blocks, int blocksize) {
  setup_kernel<<<blocks, blocksize>>>(state, std::time(nullptr));
}

//code thanks to https://gist.github.com/NicholasShatokhin/3769635
__global__
void setup_kernel(cuSt *state, ulong seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}

__device__
void generate_gpu(cuSt* globalState, double *out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    cuSt localState = globalState[idx];
    *out = curand_uniform(&localState) * M_PIl * 2; //random angle, in rad
    globalState[idx] = localState;
}

void generate_cpu(double *out)
{
  *out = std::rand() * M_PIl * 2; //random angle, in rad
}

__host__ __device__
void generate_any(cuSt *globalState, double *out) {
#ifdef __CUDA_ARCH__
  generate_gpu(globalState, out);
#else
  generate_cpu(out);
#endif
}
