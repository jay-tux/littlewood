#include "tests.h"

#define CLEANUP_TEST() free(roots); free(coeff);
#define SETUP_TEST(degree) offset = 0; roots = (complex *)malloc(degree * sizeof(complex)); coeff = (double *)malloc((degree + 1) * sizeof(double));
#define PRINT_SOL(deg) for(int i = 0; i < deg; i++) std::cout << " -> Solution: " << roots[i] << std::endl;

void check(double *coeff, int degree, complex root) {
  complex rootvalue = { 0.0, 0.0 };
  complex curr = { 1.0, 0.0 };
  for(int i = 0; i <= degree; i++) {
    rootvalue = rootvalue + (curr * coeff[i]);
    curr = curr * root;
  }
  complex zero = { 0.0, 0.0 };
  EQUAL(rootvalue, zero);
}

IMPLEMENTTEST(bairstow) {
  double  guesses[2] = { -1.0, 1.0 }; //always
  double  *coeff = nullptr;
  complex solutions[] = {
    //degree 0 (offset 0)
    //degree 1 (offset 0)
    { -7.0 / 3.0,        0.0 },
    //degree 2.1 (offset 1)
    {  4.0,              0.0 },
    { -2.0 / 3.0,        0.0 },
    //degree 2.2 (offset 3)
    { -7.0 / 2.0,  1.0 / 2.0 },
    { -7.0 / 2.0, -1.0 / 2.0 },
    //degree 3.1 (offset 5)
    { -7.0 / 2.0, -1.0 / 2.0 },
    { -7.0 / 2.0,  1.0 / 2.0 },
    {  7.0,              0.0 },
    //degree 3.2 (offset 8)
    {  3.0,              0.0 },
    {  5.0,              0.0 },
    { -7.0,              0.0 },
    //degree 4.1 (offset 11)
    {  0.1,             -0.3 },
    {  0.1,              0.3 },
    {  1.0 / 2.0,  1.0 / 2.0 },
    {  1.0 / 2.0, -1.0 / 2.0 },
    //degree 4.2 (offset 15)
    { -3.0,              0.0 },
    {  5.0 / 2.0,        0.0 },
    {  7.0,              2.0 },
    {  7.0,             -2.0 },
    //degree 4.3 (offset 19)
    { -4.0,              0.0 },
    {  4.0 / 3.0,        0.0 },
    {  5.0 / 7.0,        0.0 },
    {  0.0,              0.0 },
    //repeated roots 1 (offset 23)
    {  3.0,              0.0 },
    //repeated roots 2 (offset 24)
    { -2.0,              0.0 },
    {  1.0,              0.0 },
    //repeated roots 3 (offset 26)
    { -2.0 / 3.0,        0.0 }
  };
  complex *roots = nullptr;
  uint     offset;
  setup_cpu();

  SUBTEST(base case 1: degree 0);
  coeff = (double *)malloc(sizeof(double));
  offset = 0;
  *coeff = -1.0;
  //shouldn't crash
  bairstow(coeff, 0, guesses, roots, 0, &offset, nullptr);
  free(coeff);

  SUBTEST(base case 2: degree 1);
  SETUP_TEST(1);
  coeff[0] = 7.0; coeff[1] = 3.0;
  bairstow(coeff, 1, guesses, roots, 0, &offset, nullptr);
  EQUAL(roots[0], solutions[0]);
  CLEANUP_TEST();

  SUBTEST(base case 3.1: degree 2 - only real solutions);
  SETUP_TEST(2);
  coeff[0] = -8.0; coeff[1] = -10; coeff[2] = 3;
  bairstow(coeff, 2, guesses, roots, 0, &offset, nullptr);
  EQUAL(roots[0], solutions[1]);
  EQUAL(roots[1], solutions[2]);
  CLEANUP_TEST();

  SUBTEST(base case 3.2: degree 2 - only complex solutions);
  SETUP_TEST(2);
  coeff[0] = 25.0; coeff[1] = 14.0; coeff[2] = 2.0;
  bairstow(coeff, 2, guesses, roots, 0, &offset, nullptr);
  EQUAL(roots[0], solutions[3]);
  EQUAL(roots[1], solutions[4]);
  CLEANUP_TEST();

  SUBTEST(recursive case 1.1: degree 3 - 2 complex & 1 real solution);
  SETUP_TEST(3);
  coeff[0] = -175.0; coeff[1] = -73.0; coeff[2] = 0.0; coeff[3] = 2.0;
  bairstow(coeff, 3, guesses, roots, 0, &offset, nullptr);
  EQUAL(roots[0], solutions[5]);
  EQUAL(roots[1], solutions[6]);
  EQUAL(roots[2], solutions[7]);
  CLEANUP_TEST();

  SUBTEST(recursive case 1.2: degree 3 - 3 real solutions);
  SETUP_TEST(3);
  coeff[0] = 105.0; coeff[1] = -41.0; coeff[2] = -1.0; coeff[3] = 1.0;
  bairstow(coeff, 3, guesses, roots, 0, &offset, nullptr);
  PRINT_SOL(3);
  EQUAL(roots[0], solutions[8]);
  EQUAL(roots[1], solutions[9]);
  EQUAL(roots[2], solutions[10]);
  CLEANUP_TEST();

  SUBTEST(recursive case 2.1: degree 4 - 2 pairs of complex solutions);
  SETUP_TEST(4);
  coeff[0] = 1.0/2.0; coeff[1] = -2.0; coeff[2] = 8.0; coeff[3] = -12.0; coeff[4] = 10.0;
  bairstow(coeff, 4, guesses, roots, 0, &offset, nullptr);
  EQUAL(roots[0], solutions[11]);
  EQUAL(roots[1], solutions[12]);
  EQUAL(roots[2], solutions[13]);
  EQUAL(roots[3], solutions[14]);
  CLEANUP_TEST();

  SUBTEST(recursive case 2.2: degree 4 - 2 real & 2 complex solutions);
  SETUP_TEST(4);
  coeff[0] = -795.0; coeff[1] = 263.0; coeff[2] = 77.0; coeff[3] = -27.0; coeff[4] = 2.0;
  bairstow(coeff, 4, guesses, roots, 0, &offset, nullptr);
  EQUAL(roots[0], solutions[15]);
  EQUAL(roots[1], solutions[16]);
  EQUAL(roots[2], solutions[17]);
  EQUAL(roots[3], solutions[18]);
  CLEANUP_TEST();

  SUBTEST(recursive case 2.3: degree 4 - 4 real solutions);
  SETUP_TEST(4);
  coeff[0] = 0.0; coeff[1] = 320.0; coeff[2] = -608.0; coeff[3] = 164.0; coeff[4] = 84.0;
  bairstow(coeff, 4, guesses, roots, 0, &offset, nullptr);
  EQUAL(roots[0], solutions[19]);
  EQUAL(roots[1], solutions[20]);
  EQUAL(roots[2], solutions[21]);
  EQUAL(roots[3], solutions[22]);
  CLEANUP_TEST();

  SUBTEST(large degree);
  SETUP_TEST(6);
  coeff[0]  = -1.0; coeff[1]  =  1.0; coeff[2]  = -1.0; coeff[3]  =  1.0;
  coeff[4]  = -1.0; coeff[5]  = -1.0; coeff[6]  =  1.0; /*coeff[7]  =  1.0;
  coeff[8]  =  1.0; coeff[9]  =  1.0; coeff[10] =  1.0; coeff[11] =  1.0;
  coeff[12] =  1.0; coeff[13] =  1.0; coeff[14] =  1.0; coeff[15] =  1.0;*/
  bairstow(coeff, 6, guesses, roots, 0, &offset, nullptr);
  PRINT_SOL(6);
  CLEANUP_TEST();


  SUCCEED()
}
