#ifndef _JAY_TESTSUITE
#define _JAY_TESTSUITE

#include <csignal>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <string>

//suite namespace; you shouldn't 'using' this namespace.
namespace jay::tests::suite {

//signal to send when a test fails (shouldn't be SIGINT or SIGSEGV)
#ifndef _JAY_TESTSUITE_ERRORSIG
  #define _JAY_TESTSUITE_ERRORSIG SIGABRT
#endif //abort-on-fail

//set up test suite (should only be called once)
#define INITTEST() { std::signal(SIGINT, jay::tests::suite::h_crash); std::signal(_JAY_TESTSUITE_ERRORSIG, jay::tests::suite::h_failed); std::signal(SIGSEGV, jay::tests::suite::h_crash); }
//print a finishing message (should only be called once)
#define TESTING_FINISHED() { std::cout << "\t\t+----------------+" << std::endl << "\t\t|    ALL TESTS   |" << std::endl << "\t\t|    SUCCEEDED   |" << std::endl << "\t\t+----------------+" << std::endl; }

//defines a test with name
#define DEFINETEST(name) int test_##name();
//implements a test with name
#define IMPLEMENTTEST(name) int test_##name()
//runs a named test (behavior is dependent on settings)
#define RUNTEST(name) { std::cout << "\t====== RUNNING TEST " << #name << " ======" << std::endl; int res = test_##name(); if(res) { std::cout << "\t====== TEST " << #name << " SUCCEEDED ======" << std::endl; } }
//makes a test succeed
#define SUCCEED() { return 1; }

//helper macros
#define ABS(val) ((val) < 0 ? -(val) : (val))
#define FAIL(cond) { std::cerr << cond << std::endl; std::raise(_JAY_TESTSUITE_ERRORSIG); }
#define WARNEQUAL(v1, v2) "Expected " << #v1 << "(" << v1 << ")" << " and " << #v2 << "(" << v2 << ")" << "to be equal"
#define WARNDIFF(v1, v2) "Expected " << #v1 << "(" << v1 << ")" << " and " << #v2 << "(" << v2 << ")" << "to be different"
#define WARNLESS(v1, v2) "Expected " << #v1 << "(" << v1 << ")" << " to be less than " << #v2 << "(" << v2 << ")"
#define WARNGREATER(v1, v2) "Expected " << #v1 << "(" << v1 << ")" << " to be greater than " << #v2 << "(" << v2 << ")"
#define WARNLEQ(v1, v2) "Expected " << #v1 << "(" << v1 << ")" << " to be less than or equal to " << #v2 << "(" << v2 << ")"
#define WARNGEQ(v1, v2) "Expected " << #v1 << "(" << v1 << ")" << " to be greater than equal to " << #v2 << "(" << v2 << ")"
#define WARNNULL(v) "Expected " << #v << " to be null, points to " << static_cast<void *>(v)
#define WARNNONNULL(v) "Expected " << #v << " to be non-null"
#define WARNFALSE(v) "Expected " << #v << " to be true, is false"
#define WARNTRUE(v) "Expected " << #v << " to be false, is true"
#define WARNFAR(v1, v2, tol, diff) "Expected " << #v1 << "(" << v1 << ") and " << #v2 << "(" << v2 << ") to be within " << tol << " (is " << diff << ")"
#define WARNCLOSE(v1, v2, tol, diff) "Expected " << #v1 << "(" << v1 << ") and " << #v2 << "(" << v2 << ") to be more than " << tol << " apart (is " << diff << ")"

//create a subtest (behavior depends on settings)
#define SUBTEST(name) { std::cout << "=== SUBTEST " << #name << " ===" << std::endl; }

//checks if two variables/values are equal (using operator)
#define EQUAL(v1, v2) if(!(v1 == v2)) { FAIL(WARNEQUAL(v1, v2)) }
//checks if two variables/values are not equal (using operator)
#define NOTEQUAL(v1, v2) if(!(v1 != v2)) { FAIL(WARNDIFF(v1, v2)) }
//checks if a variable/value is less than another variable/value (using operator)
#define LESS(v1, v2) if(!(v1 < v2)) { FAIL(WARNLESS(v1, v2)) }
//checks if a variable/value is greater than another variable/value (using operator)
#define GREATER(v1, v2) if(v1 <= v2) { FAIL(WARNGREATER(v1, v2)) }
//checks if a variable/value is less than or equal to another variable/value (using operator)
#define LEQ(v1, v2) if(!(v1 <= v2)) { FAIL(WARNLEQ(v1, v2)) }
//checks if a variable/value is greater than or equal to another variable/value (using operator)
#define GEQ(v1, v2) if(!(v1 >= v2)) { FAIL(WARNGEQ(v1, v2)) }
//checks if a variable (pointer) is not null
#define NONNULL(v) if((void *)v == (void *)0 || v == nullptr) { FAIL(WARNNONNULL(v)) }
//checks if a variable (pointer) is null
#define CHECKNULL(v) if((void *)v != (void *)0 || v != nullptr) { FAIL(WARNNULL(v)) }
//checks if a variable/value is true
#define CHECKTRUE(v) if(!v) { FAIL(WARNFALSE(v)) }
//checks if a variable/value is false
#define CHECKFALSE(v) if(v) { FAIL(WARNTRUE(v)) }
//checks if two variables are "close" together
#define APPROX(v1, v2, tol) if(ABS(v1 - v2) > tol) { FAIL(WARNFAR(v1, v2, tol, ABS(v1 - v2))) }
//checks if two variable aren't "close" together
#define NAPPROX(v1, v2, tol) if(ABS(v1 - v2) <= tol) { FAIL(WARNCLOSE(v1, v2, tol, ABS(v1 - v2))) }

//handles a program interrupt (SIGINT/SIGSEGV/...)
void h_crash(int signal);
//handles a failed test
void h_failed(int signal);

} //namespace
#endif
