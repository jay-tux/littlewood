#include "profiler.h"

#if PROFILING
//<editor-fold> Profiler
Profiler Profiler::instance = Profiler();
Profiler::Profiler() {}

void Profiler::add(int degree, proftime start, proftime init, proftime thread, proftime sync, proftime end) {
  proflog add = {
    degree,                   //degree
    NANO_DIFF(start, init),   //time to init polynomials, alloc memory
    NANO_DIFF(init, thread),  //time to start all the threads
    NANO_DIFF(thread, sync),  //time to sync the gpu (finish execution)
    NANO_DIFF(sync, end)      //time to write data
  };
  this->logs.push_back(add);
}

void Profiler::write(std::string file) {
  std::ofstream output;
  output.open(file, std::ofstream::app);
  output << "degree,initialization time(ns),thread starting time(ns),"
    << "actual run time(ns),output write time(ns)" << std::endl;
  for(auto log : this->logs) {
    output << log.degree << "," << log.init.count() << ","
      << log.threadstart.count() << "," << log.synced.count() << ","
      << log.write.count() << std::endl;
  }
}
//</editor-fold>
#endif
