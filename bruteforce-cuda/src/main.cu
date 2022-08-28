#include <device_launch_parameters.h>
#include <iostream>
#include <chrono>
#include <stdlib.h>
#include "macros.h"
#include "bairstow.h"
#include "profiler.h"
#include "rng.h"

using namespace std::literals;

int main(int argc, char **argv)
{
  if(argc == 2 && std::string(argv[1]) == "-h") {
    std::cout << " ==== Littlewood Roots (CUDA) === " << std::endl;
#ifdef PROFILING
    std::cout << " Usage: " << argv[0] << " <max degree> <log file>" << std::endl;
#else
    std::cout << " Usage: " << argv[0] << " <max degree>" << std::endl;
#endif
    std::cout << " Set <max degree> equal to -1 to keep running until interrupted" << std::endl;
  }

  int max;
#ifdef PROFILING
  if(argc > 2) {
#else
  if(argc > 1) {
#endif
    max = std::atoi(argv[1]);
  }
  else {
    std::cerr << "Degree: ";
    std::cin >> max;
    auto start = NANO_NOW;
    auto sol = littlewood(max);
    auto solved = NANO_NOW;
    writeRoots(sol);
    auto end = NANO_NOW;
    std::cerr << "\tTotal time:   " << NANO_PRINT(start, end) << "ms" << std::endl
              << "\tSolving time: " << NANO_PRINT(start, solved) << "ms" << std::endl
              << "\tWriting time: " << NANO_PRINT(solved, end) << "ms" << std::endl;
    return 0;
  }

  setup_cpu();
  for(uint i = 1; i < max || max == -1; i++) {
    std::cerr << "Degree " << i << std::endl
              << "-----------" << std::endl;
    auto start = NANO_NOW;
    auto sol = littlewood(i);
    auto solved = NANO_NOW;
    writeRoots(sol);
    auto end = NANO_NOW;
    std::cerr << "\tTotal time:   " << NANO_PRINT(start, end) << "ms" << std::endl
              << "\tSolving time: " << NANO_PRINT(start, solved) << "ms" << std::endl
              << "\tWriting time: " << NANO_PRINT(solved, end) << "ms" << std::endl;
#ifdef PROFILING
    Profiler::instance.write(std::string(argv[2]));
#endif
  }


  return 0;
}
