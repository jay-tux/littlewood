#include "tests.h"
#include <sstream>

IMPLEMENTTEST(complex) {
  SUBTEST(helpers)
  complex c1 = { 1.5, 0.0 };
  complex c2 = { 3.2, -1.2 };
  complex c3 = { -0.4, 0.0 };

  EQUAL(c1.getReal(), 1.5);
  EQUAL(c2.getReal(), 3.2);
  EQUAL(c3.getReal(), -0.4);
  EQUAL(c1.getImag(), 0.0);
  EQUAL(c2.getImag(), -1.2);
  EQUAL(c3.getImag(), 0.0);

  SUBTEST(equality)
  c1 = { 2.5, 5.9 };
  c2 = { 2.5, 5.9 };
  c3 = { 8.1, 5.9 };
  complex c4 = { 8.1, 0.5 };
  complex c5 = { 2.5, 5.1 };

  EQUAL(c1, c2);
  NOTEQUAL(c1, c3);
  NOTEQUAL(c1, c4);
  NOTEQUAL(c1, c5);

  SUBTEST(solve ax+b=0)
  fromFirst(5.0, 8.0, &c1); //5x + 8 = 0
  fromFirst(7.0, -2.0, &c2); //7x - 2 = 0
  fromFirst(1.0, 0.0, &c3); //x = 0

  APPROX(c1.r, -8.0 / 5.0, TOLERANCE);
  EQUAL(c1.i, 0.0)
  APPROX(c2.r, 2.0 / 7.0, TOLERANCE);
  EQUAL(c2.i, 0.0);
  EQUAL(c3.r, 0.0);
  EQUAL(c3.i, 0.0);

  SUBTEST(solve ax^2 + bx + c)
  fromQuadr(5.0, 0.0, 0.0, &c1, &c2); //5x^2 = 0
  EQUAL(c1.r, 0.0); EQUAL(c1.i, 0.0);
  EQUAL(c2.r, 0.0); EQUAL(c2.i, 0.0);

  fromQuadr(3.0, 5.0, 2.0, &c1, &c2); //3x^2 + 5x + 2
  APPROX(c1.r, -2.0/3.0, TOLERANCE); EQUAL(c1.i, 0.0);
  APPROX(c2.r, -1.0, TOLERANCE); EQUAL(c2.i, 0.0);

  fromQuadr(41.0, 5.0, 1.0/4.0, &c1, &c2); //41x^2 + 5x + 1/4 = 0
  APPROX(c1.r, -5.0/82.0, TOLERANCE); APPROX(c1.i, 2.0/41.0, TOLERANCE);
  APPROX(c2.r, -5.0/82.0, TOLERANCE); APPROX(c2.i, -2.0/41.0, TOLERANCE);

  SUCCEED()
}
