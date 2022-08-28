#include "tests.h"

#define MAX_TEST_DEGREE 7

void printPolys(double *polys, uint degree, ulong arrcount) {
  for(uint po = 0; po < arrcount; po++) {
    std::cout << polys[po * degree + po];
    for(uint term = 1; term <= degree; term++) {
      std::cout << " + " << polys[po * degree + po + term] << "x^" << term;
    }
    std::cout << std::endl;
  }
}

IMPLEMENTTEST(generator) {
  for(int degree = 1; degree <= MAX_TEST_DEGREE; degree++) {
    SUBTEST(generator next degree)
    double *polys;
    ulong arrcount = 1 << degree; //(1 << (degree + 1)) / 2
    cudaMallocManaged(&polys, arrcount * (degree + 1) * sizeof(double));
    genArray<<<blockCount(arrcount), BLOCKSIZE>>>(polys, degree);
    cudaDeviceSynchronize();

    //all should be 1 or -1
    bool allvalid = true;
    for(int p = 0; p < (arrcount) * (degree + 1); p++) {
      if(polys[p] != 1.0 && polys[p] != -1.0) { allvalid = false; break; }
    }

    //printPolys(polys, degree, arrcount);
    CHECKTRUE(allvalid);

    for(int p = 0; p < arrcount; p++) {
      EQUAL(polys[p * (degree + 1) + degree], 1.0);
    }
  }
  SUCCEED()
}

#undef MAX_TEST_DEGREE
