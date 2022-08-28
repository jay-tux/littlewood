CC=nvcc
CCARGS=-g -I. -I/opt/cuda/samples/common/inc/ -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored -c
LDARGS=-lGL -lGLEW -lglfw -lX11 -lpthread -lXrandr -lXi -ldl
OBJECTS=obj/main.o obj/lib/glad.o obj/shader.o obj/camera.o obj/setup.o obj/cuda/setup.o obj/cuda/kernel.o obj/cuda/gldebug.o

all: gl

gl: $(OBJECTS)
	$(CC) $(LDARGS) $(OBJECTS) -o bin/gl

cpu: src/cpu.cpp
	$(CC) $< -o bin/cpu

obj/cuda/%.o: src/cuda/%.cu src/cuda/%.h Makefile
	$(CC) $(CCARGS) $< -o $@

obj/%.o: src/%.cpp glad/include/glad/glad.h src/%.h Makefile
	$(CC) $(CCARGS) $< -o $@

obj/lib/glad.o: glad/src/glad.c glad/include/glad/glad.h
	$(CC) $(CCARGS) $< -o $@

clear:
	rm -f bin/*
	rm -rf obj/*
	mkdir obj/lib
	mkdir obj/cuda

.PHONY: all clean cpu
