#include "complex.h"

double complex::tolerance = 1e-10;

double complex::getReal() { return this->r; }
double complex::getImag() { return this->i; }
bool complex::operator<(const complex &other) const { return this->r < other.i; }

bool complex::operator==(const complex &other) const {
  double diffr = this->r - other.r;
  double diffi = this->i - other.i;
  return (diffr * diffr + diffi * diffi) < complex::tolerance;
}

bool complex::operator!=(const complex &other) const {
  return !(*this == other);
}

complex complex::operator*(const complex &other) const {
  return { this->r * other.r - this->i * other.i, this->r * other.i + this->i * other.r };
}

complex complex::operator*(double other) const {
  return { this->r * other, this->i * other };
}

complex complex::operator+(const complex &other) const {
  return { this->r + other.r, this->i + other.i };
}

std::ostream &operator<<(std::ostream &s, const complex &c)
{
  /*
      Formatting:
       - { 10,  4 } -> 1.00e1 4.00e0
       - { -1,  4 } -> -1.00e0 4.00e0
       - { 10, -4 } -> 1.00e1 -4.00e0
       - { -1, -4 } -> -1.00e0 -4.00e0
  */
  return (s << std::scientific << c.r << " " << c.i);
}

__host__ __device__ void fromFirst(double a, double b, complex *ans)
{
#ifdef __CUDA_ARCH__
  //printf("Thread %d: Writing { %+e, %+e } to %p\n", GSLOOP_INDEX, (-b/a), 0.0, ans);
#endif
  *ans = { (-b/a), 0.0 };
}

__host__ __device__ void fromQuadr(double a, double b, double c, complex *ans1, complex *ans2)
{
  double d = b*b - 4*a*c;
  if(d >= 0) {
    //double nu = std::sqrt(b);
    *ans1 = { ( std::sqrt(d) - b) / (2*a), 0.0f };
    *ans2 = { (-std::sqrt(d) - b) / (2*a), 0.0f };
  }
  else {
    *ans1 = { -b / (2*a),  std::sqrt(-d) / (2*a) };
    *ans2 = { -b / (2*a), -std::sqrt(-d) / (2*a) };
  }
#ifdef __CUDA_ARCH__
  /*printf("Thread %d: Writing { %+e, %+e } to %p\n", GSLOOP_INDEX, ans1->r, ans1->i, ans1);
  printf("Thread %d: Writing { %+e, %+e } to %p\n", GSLOOP_INDEX, ans2->r, ans2->i, ans2);*/
#endif
}
