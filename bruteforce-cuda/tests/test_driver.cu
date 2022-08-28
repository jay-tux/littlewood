#include "tests.h"

int main() {
  INITTEST();
  RUNTEST(complex);
  RUNTEST(generator);
  RUNTEST(bairstow);
  RUNTEST(littlewood);

  TESTING_FINISHED();
}
