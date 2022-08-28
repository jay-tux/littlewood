#ifndef _JAY_CUDA_COMPLEX
#define _JAY_CUDA_COMPLEX

#include <iostream>
#include "macros.h"

//complex numbers
struct complex {
  //the real part
  double r;
  //the imaginary part
  double i;

  //get real part of complex
  double getReal();
  //get imaginary part of complex
  double getImag();

  //equality tolerance
  static double tolerance;

  //comparison !!no comparison over complex, only for std::set!!
  bool operator<(const complex &other) const;
  //checks whether two complex numbers are (approximately) equal.
  bool operator==(const complex &other) const;
  //checks whether two complex numbers are not (approximately) equal.
  bool operator!=(const complex &other) const;
  //multiplies two complex numbers (for checking)
  complex operator*(const complex &other) const;
  //multiplies a complex and a real number (for checking)
  complex operator*(double other) const;
  //adds two complex numbers (for checking)
  complex operator+(const complex &other) const;
};

//streams this complex number to an output stream
std::ostream &operator<<(std::ostream &s, const complex &c);
//solves an equation of the form a*x + b
__host__ __device__ void fromFirst(double a, double b, complex *ans);
//solves an equation of the form a*x^2 + b*x + c
__host__ __device__ void fromQuadr(double a, double b, double c, complex *ans1, complex *ans2);

#endif
