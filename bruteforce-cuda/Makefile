CC=nvcc
LDARGS=
CCARGS=-c -dc -DPROFILING -g -O0
TESTARGS=-Isrc/
OBJECTS=obj/bairstow.o obj/rng.o obj/complex.o obj/profiler.o
TESTOBJ=obj/test/test_complex.o obj/test/test_driver.o obj/test/test_generator.o obj/test/test_bairstow.o obj/test/test_littlewood.o

all: cuda

cuda: $(OBJECTS) obj/main.o
	@echo "--- LINKING STAGE (CUDA) ---"
	$(CC) $(LDARGS) $(OBJECTS) obj/main.o -o bin/bruteforce

test: $(OBJECTS) $(TESTOBJ) obj/test/suite.o
	@echo "--- LINKING TESTS ---"
	$(CC) $(LDARGS) $(TESTOBJ) obj/test/suite.o $(OBJECTS) -Isrc/ -o bin/test
	bin/test

obj/%.o: src/%.cu src/macros.h Makefile
	$(CC) $(CCARGS) $(LDARGS) $< -o $@

obj/test/suite.o: tests/testsuite.cpp tests/testsuite.h
	$(CC) $(CCARGS) $(LDARGS) $< -o $@

obj/test/%.o: tests/%.cu tests/tests.h tests/testsuite.h
	$(CC) $(CCARGS) $(LDARGS) $(TESTARGS) $< -o $@

clear:
	rm -f bin/*
	rm -rf obj/*
	mkdir obj/test

.PHONY: all clear test cuda
