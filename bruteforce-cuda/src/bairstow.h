#ifndef _JAY_CUDA_BAIRSTOW
#define _JAY_CUDA_BAIRSTOW

#include <iostream>
#include <fstream>
#include <string>
#include <set>
#include <vector>
#include <cmath>
#include "macros.h"
#include "rng.h"
#include "complex.h"
#include "profiler.h"

#ifndef TOLERANCE
#define TOLERANCE 1e-15
#endif

#ifndef INF_TOL
#define INF_TOL 1e+19
#endif

#ifndef MAX_TRIES
#define MAX_TRIES 200
#endif

#ifndef MAX_RESTARTS
#define MAX_RESTARTS (1 << 20)
#endif

#ifndef ABS
  #define ABS(x) ((x) < 0 ? (-(x)) : (x))
#endif

//get block count (at least certain number of threads started)
int blockCount(double threadsRequired);

//[GPU] run bairstow algorithm (can also be run on the CPU, useful for debugging).
//        coeff are the coefficients, where coeff[0] is the constant term
//        degree is the degree of the polynomial, so coeff has degree + 1 values
//        guesses is a 2-element array containing two initial guesses for the quadratic equation
//        roots is the output array, will be overwritten (requires at least degree free spaces)
//        index is the starting index in the roots array
//        offset is the offset seen from the starting index
__device__ __host__ void bairstow(double *coeff, uint degree, double *guesses, complex *roots, ulong index, uint *offset, cuSt *state);
//[GPU] start the bairstow algorithm (most arguments are passed to the bairstow unchanged).
//        coeff is modified: the right starting index is calculated from the thread id
//        starting index for the solutions is generated from the thread id as well
//        guesses are per-thread generated
__global__ void start_bairstow(double *coeff, uint degree, complex *roots, cuSt *state);
//[CPU] generate all combinations for littlewood polynomials of certain degree
//        arrays contains 2^(degree + 1) arrays of length (degree + 1)
void combine(double *arrays, uint degree);
//[GPU] generate all combinations for littlewood polynomials of certain degree
//        should be used in a grid-stride loop; otherwise same to combine
__global__ void genArray(double *target, uint degree);
//[CPU->GPU] get all littlewood roots for a certain degree, write them to outfile
std::vector<complex> littlewood(uint degree);
//[CPU] writes roots (from degree) to stdout
void writeRoots(std::vector<complex> roots);

#endif
