#include "testsuite.h"

void jay::tests::suite::h_crash(int signal) {
  std::cerr << "+---------------------+" << std::endl
            << "| ERROR WHILE TESTING |" << std::endl
            << "+---------------------+" << std::endl
            << "  -> the environment threw signal " << signal << std::endl;
  std::exit(0);
}

void jay::tests::suite::h_failed(int signal) {
  std::cerr << "+-------------------------+" << std::endl
            << "|    TESTS INTERRUPTED    |" << std::endl
            << "+-------------------------+" << std::endl
            << "    ";
  switch(signal) {
    case _JAY_TESTSUITE_ERRORSIG: std::cerr << "failed test";        break;
    case SIGINT:                  std::cerr << "user interrupt";     break;
    case SIGSEGV:                 std::cerr << "segmentation fault"; break;
    default:                      std::cerr << "unknown cause";      break;
  }
  std::cerr << std::endl;
  std::exit(0);
}
