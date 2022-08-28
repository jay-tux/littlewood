#ifndef _JAY_CUDA_KERNEL
#define _JAY_CUDA_KERNEL

#include "setup.h"
#include <iostream>
#include <iomanip>
#include <cuComplex.h>

#define re(z)      (z.x)
#define im(z)      (z.y)
#define cadd(a, b) (complex(a.x + b.x,              a.y + b.y))
#define csub(a, b) (complex(a.x - b.x,              a.y - b.y))
#define cmul(a, b) (complex(a.x * b.x - a.y * b.y,  a.x * b.y + a.y * b.x))
#define cinv(a)    (complex(a.x / snorm(a),   -a.y / snorm(a)))
#define norm(a)    (a.x * a.x + a.y * a.y)
#define snorm(a)   (sqrt(norm(z)))

#define ccomp(r, i) (make_cuComplex(r, i))
#define ccompf(r)   (make_cuComplex(r, 0.0f))

struct complex {
  __host__ __device__ complex() : x{0.0f}, y{0.0f} {}
  __host__ __device__ complex(float re, float im) : x{re}, y{im} {}
  float x, y;
};

struct transform {
  complex center;
  float zoom;
  bool hasChanged;
};

struct transformer {
  bool up, down, left, right, in, out, reset;
  __inline__ bool changed() const;
};

__device__ cuComplex operator+(cuComplex a, cuComplex b);
__device__ cuComplex operator-(cuComplex a, cuComplex b);
__device__ cuComplex operator*(cuComplex a, cuComplex b);
__device__ cuComplex operator/(cuComplex a, cuComplex b);
__device__ cuComplex operator+(cuComplex a, float     b);
__device__ cuComplex operator-(cuComplex a, float     b);
__device__ cuComplex operator*(cuComplex a, float     b);
__device__ cuComplex operator/(cuComplex a, float     b);
__device__ cuComplex operator+(float     a, cuComplex b);
__device__ cuComplex operator-(float     a, cuComplex b);
__device__ cuComplex operator*(float     a, cuComplex b);
__device__ cuComplex operator/(float     a, cuComplex b);
__device__ float     operator!(cuComplex a); //norm operator
__device__ cuComplex operator~(cuComplex a); //reciprocal operator
__device__ cuComplex toComplex(complex &pix, complex &bounds, transform &t);

__device__ bool isRoot(cuComplex z, float tol, int *it, int maxit);
__global__ void kernel(float *buf, float *itbuf, uint w, uint h, transform t);
void callKernel(cuData &data, cuGL &buf, transform &t);

void recenter(transformer &delta, transform &t);

#endif
