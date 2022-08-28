#include <chrono>
#include <iomanip>
#include <cmath>

#define BLOCKSIZE 256
#define GSLOOP_INDEX (blockIdx.x * blockDim.x + threadIdx.x)
#define GSLOOP_STRIDE (blockDim.x * gridDim.x)
#define GSLOOP(indexer, max) for(int indexer = GSLOOP_INDEX; indexer < max; indexer += GSLOOP_STRIDE)
#define GSLOUTOFBOUNDS(max) if(GSLOOP_INDEX >= (max)) return;

#define NANO_NOW (std::chrono::steady_clock::now())
#define NANO_DIFF(start, end) (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start))
#define NANO_PRINT(start, end) (NANO_DIFF(start, end) / 1ms)

#define MINDELTA 1e-10
#define CUDA_INFNAN(x) (isnan(x) || isinf(x))
#define CPP_INFNAN(x) (!std::isfinite(x))
#define NOCHANGE(x, y) (ABS(x - y) < MINDELTA)

#define POLYOFFSET(i, degree) (i * (degree + 1))
#define SOLOFFSET(i, degree)  (i * degree)
