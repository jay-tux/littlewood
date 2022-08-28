#include "bairstow.h"

//<editor-fold> complex/helpers
int blockCount(double threadsRequired)
{
  int cnt = threadsRequired / BLOCKSIZE;
  if(cnt * BLOCKSIZE < threadsRequired) return cnt+1;
  return cnt;
}
//</editor-fold>

//<editor-fold> bairstow algorithm
__global__
void start_bairstow(double *coeff, uint deg, complex *roots, cuSt *state) {
  GSLOUTOFBOUNDS((1 << deg));

  uint   *offset   = (uint *)  malloc(    sizeof(uint));
  double *guess    = (double *)malloc(2 * sizeof(double));
  guess[0] = (deg > 0) ? (coeff[deg - 1] / coeff[deg]) : (1.0);
  guess[1] = (deg > 1) ? (coeff[deg - 2] / coeff[deg]) : (1.0);

  double *modcoeff = coeff + (GSLOOP_INDEX * (deg + 1)); //go to start of this polynomial
  ulong   ind      = GSLOOP_INDEX * deg;
  *offset = 0;

  bairstow(modcoeff, deg, guess, roots, ind, offset, state);
  free(offset);
  free(guess);
}

__device__ __host__
void bairstow(double *coeff, uint deg, double *guess, complex *roots, ulong ind, uint *offset, cuSt *state) {
  if(deg < 1) { return; } //1st base case: not a polynomial

  if(deg == 1) { //2nd base case: ax+b
    fromFirst(coeff[1], coeff[0], roots + ind + *offset); //&roots[ind + *offset]
    *offset = *offset + 1;
#ifdef __CUDA_ARCH__
    //printf("THREAD %d: finished.\n", GSLOOP_INDEX);
#endif
    return;
  }

  if(deg == 2) { //3rd base case: ax^2 + bx + c
    fromQuadr(coeff[2], coeff[1], coeff[0],
      roots + ind + *offset, roots + ind + *offset + 1); //&roots[ind + *offset] & after
    *offset = *offset + 2;
#ifdef __CUDA_ARCH__
    //printf("THREAD %d: finished.\n", GSLOOP_INDEX);
#endif
    return;
  }

  int n = deg;
  double *b = (double *)malloc((n + 1) * sizeof(double));
  double *c = (double *)malloc((n + 1) * sizeof(double));
  long tries = 0;
  double r = guess[0]; double s = guess[1];
  long restart = 0;

  do {
#ifdef __CUDA_ARCH__
    if(tries >= MAX_TRIES || CUDA_INFNAN(guess[0]) || CUDA_INFNAN(guess[1])
      || NOCHANGE(guess[0], r) || NOCHANGE(guess[1], s))
#else
    if(tries >= MAX_TRIES || CPP_INFNAN(guess[0]) || CPP_INFNAN(guess[1])
      || NOCHANGE(guess[0], r) || NOCHANGE(guess[1], s))
#endif
    {
      double angle;
      double rad = 0.0;
      for(int i = 0; i < deg; i++) rad += ABS(coeff[i]);
      generate_any(state, &angle);
      guess[0] = rad * rad * cos(angle);
      guess[1] = rad * rad * sin(angle);
      tries = 0;
      restart++;

      if(restart >= MAX_RESTARTS) {
        //cancel operation
        for(int i = 0; i < deg; i++) roots[ind + *offset + i] = { 0.0, 0.0 };
#ifdef __CUDA_ARCH__
        //printf("THREAD %d interrupted at degree %d.\n", GSLOOP_INDEX, deg);
#endif
        return;
      }
    }

    r = guess[0]; s = guess[1];
    b[n] = coeff[n]; b[n - 1] = coeff[n - 1] + guess[0] * b[n];
    c[n] =     b[n]; c[n - 1] =     b[n - 1] + guess[0] * c[n];

    for(int i = n - 2; i >= 0; i--)
      b[i] = coeff[i] + guess[0] * b[i + 1] + guess[1] * b[i + 2];

    for(int i = n - 2; i >= 1; i--)
      c[i] =     b[i] + guess[0] * c[i + 1] + guess[1] * c[i + 2];

    double din = 1.0 / (c[2] * c[2] - c[1] * c[3]);
    guess[0] = guess[0] + din * (-b[1] * c[2] + b[0] * c[3]);
    guess[1] = guess[1] + din * ( b[1] * c[1] - b[0] * c[2]);

    tries++;
  } while(ABS(b[0]) > TOLERANCE || ABS(b[1]) > TOLERANCE);

  if(deg >= 3) {
    double dis = guess[0] * guess[0] + 4 * guess[1];
    if(dis < 0) {
      roots[ind + *offset    ] = { guess[0]/2.0, -std::sqrt(-dis)/2.0 };
      roots[ind + *offset + 1] = { guess[0]/2.0,  std::sqrt(-dis)/2.0 };
    }
    else {
      roots[ind + *offset    ] = { (guess[0] - std::sqrt(dis)) / 2.0, 0.0 };
      roots[ind + *offset + 1] = { (guess[0] + std::sqrt(dis)) / 2.0, 0.0 };
    }
    *offset = *offset + 2;
    guess[0] = (deg > 2) ? (b[deg - 1] / b[deg]) : (1.0);
    guess[1] = (deg > 3) ? (b[deg - 2] / b[deg]) : (1.0);
    bairstow(b + 2, deg - 2, guess, roots, ind, offset, state); //b+2 = b[2, ...]
  }
  free(b);
  free(c);
}

//</editor-fold>

//<editor-fold> driver code
__global__
void genArray(double *target, uint degree)
{
  GSLOUTOFBOUNDS((1 << degree));

  int start = GSLOOP_INDEX * (degree + 1);
  for(int subi = 0; subi < degree; subi++) {
    target[subi+start] = (GSLOOP_INDEX & (1 << subi)) ? 1.0 : -1.0;
  }
  target[degree+start] = 1.0;
}

void combine(double *arrays, uint degree)
{
  int index = 0;
  //generate using shifting. Replace 0 with -1.0f, 1 with 1.0f
  for(int dom = 0; dom < (1 << (degree + 1)); dom++) {
    for(int sub = 0; sub <= degree; sub++) {
      arrays[index] = (dom & (1 << sub)) ? 1.0 : -1.0;
      index++;
    }
  }
}

std::vector<complex> littlewood(uint degree)
{
#if PROFILING
  auto start = NANO_NOW;
#endif
  //init
  double *polynomials;
  complex *solutions;
  cuSt *state;
  ulong arrcount = 1 << (degree);
  cudaMallocManaged(&polynomials, arrcount * (degree + 1) * sizeof(double));
  cudaMallocManaged(&solutions,   arrcount * degree       * sizeof(complex));
  cudaMallocManaged(&state,       arrcount                * sizeof(cuSt));
  setup_gpu(state, blockCount(arrcount), BLOCKSIZE); //init rngs
  genArray<<<blockCount(arrcount), BLOCKSIZE>>>(polynomials, degree);
#if PROFILING
  auto init = NANO_NOW;
#endif

  start_bairstow<<<blockCount(arrcount), BLOCKSIZE>>>(polynomials, degree, solutions, state);
#if PROFILING
  auto thread = NANO_NOW;
#endif

  //wait for device
  cudaDeviceSynchronize();
#if PROFILING
  auto synced = NANO_NOW;
#endif

  //cleanup, return value
  cudaFree(polynomials);
  std::vector<complex> res;
  res.reserve(arrcount * degree);

  for(ulong i = 0; i < arrcount; i++) {
    for(int c = 0; c < degree; c++) {
      std::cout << solutions[i * degree + c] << std::endl;
      //res.push_back(solutions[i * degree + c]);
    }
  }
#ifdef PROFILING
  auto end = NANO_NOW;
  Profiler::instance.add(degree, start, init, thread, synced, end);
#endif
  return res;
}

void writeRoots(std::vector<complex> roots)
{
  for(auto root : roots) {
    //std::cout << root << std::endl;
  }
}
//</editor-fold>
