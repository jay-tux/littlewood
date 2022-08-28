#include "tests.h"

IMPLEMENTTEST(littlewood) {
  auto printer = [](auto set) {
    for(auto &t : set) std::cout << t << std::endl;
  };

  auto is_solution = [](auto res, complex &c) {
    for(auto &in : res) {
      if(c == in) return true;
    }
    return false;
  };

  SUBTEST(degree 1)
  complex sol1[] = {
    { -1.0, 0.0 },
    {  1.0, 0.0 }
  };
  auto d1 = littlewood(1);
  CHECKTRUE(is_solution(d1, sol1[0]));
  CHECKTRUE(is_solution(d1, sol1[1]));

  SUBTEST(degree 2)
  complex sol2[] = {
    { -1.0/2.0, -0.8660254037844386467637232 },
    { -1.0/2.0,  0.8660254037844386467637232 },
    { -0.6180339887498948482045868, 0.0 },
    { -0.6180339887498948482045868, 0.0 }
  };
  auto d2 = littlewood(2);
  CHECKTRUE(is_solution(d2, sol2[0]));
  CHECKTRUE(is_solution(d2, sol2[1]));
  CHECKTRUE(is_solution(d2, sol2[2]));
  CHECKTRUE(is_solution(d2, sol2[3]));

  SUBTEST(degree 3)
  complex sol3[] = {
    { -1.0,  0.0 },
    {  1.0,  0.0 },
    {  0.0,  1.0 },
    {  0.0, -1.0 },
    {  0.5436890126920763615708560,  0.0 },
    { -0.7718445063460381807854280, -1.1151425080399373597457646 },
    { -0.7718445063460381807854280,  1.1151425080399373597457646 },
    { -0.5436890126920763615708560,  0.0 },
    {  0.7718445063460381807854280, -1.1151425080399373597457646 },
    {  0.7718445063460381807854280,  1.1151425080399373597457646 },
    { -1.839286755214161132551853 },
    {  1.839286755214161132551853 },
    {  0.4196433776070805662759263, -0.6062907292071993692593422 },
    {  0.4196433776070805662759263,  0.6062907292071993692593422 },
    { -0.4196433776070805662759263, -0.6062907292071993692593422 },
    { -0.4196433776070805662759263,  0.6062907292071993692593422 }
  };
  auto d3 = littlewood(3);
  CHECKTRUE(is_solution(d3, sol3[0]));
  CHECKTRUE(is_solution(d3, sol3[1]));
  CHECKTRUE(is_solution(d3, sol3[2]));
  CHECKTRUE(is_solution(d3, sol3[3]));
  CHECKTRUE(is_solution(d3, sol3[4]));
  CHECKTRUE(is_solution(d3, sol3[5]));
  CHECKTRUE(is_solution(d3, sol3[6]));
  CHECKTRUE(is_solution(d3, sol3[7]));
  CHECKTRUE(is_solution(d3, sol3[8]));
  CHECKTRUE(is_solution(d3, sol3[9]));
  CHECKTRUE(is_solution(d3, sol3[10]));
  CHECKTRUE(is_solution(d3, sol3[11]));
  CHECKTRUE(is_solution(d3, sol3[12]));
  CHECKTRUE(is_solution(d3, sol3[13]));
  CHECKTRUE(is_solution(d3, sol3[14]));
  CHECKTRUE(is_solution(d3, sol3[15]));
  SUCCEED()
}
