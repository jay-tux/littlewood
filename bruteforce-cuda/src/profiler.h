#ifndef _JAY_CUDA_PROFILER
#define _JAY_CUDA_PROFILER

#include <chrono>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include "macros.h"

#ifdef PROFILING
//shorthand for timestamp type
typedef std::chrono::time_point<std::chrono::steady_clock> proftime;
//shorthand for duration type
typedef std::chrono::nanoseconds profdur;

struct proflog {
  //degree of polynomial solved
  int degree;
  //durations
  profdur init, threadstart, synced, write;
};

//compiler complains
typedef std::vector<proflog> container;

//profiling log
class Profiler {
public:
  //singleton instance
  static Profiler instance;
  //add a profiling log
  void add(int deg, proftime start, proftime init, proftime thread, proftime sync, proftime end);
  //write all logs to file in CSV format
  void write(std::string file);
private:
  //singleton class, hide constructor
  Profiler();
  //profiling results
  container logs;
};
#endif //profiling

#endif
