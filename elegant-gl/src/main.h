#ifndef MAIN_H
#define MAIN_H

#include "glad/include/glad/glad.h"
#include <GLFW/glfw3.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "shader.h"
#include "camera.h"
#include "cuda/setup.h"
#include "cuda/kernel.h"
#include "setup.h"
#include "cuda/gldebug.h"

#ifndef ENABLE_CAM_ROTATION
#define ENABLE_CAM_ROTATION 0
#endif

#ifndef TEX_W
#define TEX_W 4096
#endif

#ifndef TEX_H
#define TEX_H 4096
#endif

#ifndef PROFILER
#define PROFILER 0
#endif

#if PROFILER
  #ifndef MIN_PRC
  #define MIN_PRC 1e-25
  #endif
  #ifndef MAX_PRC
  #define MAX_PRC 1e-10
  #endif
  #ifndef DELTA_PRC
  #define DELTA_PRC 2
  #endif
  #ifndef MIN_IT
  #define MIN_IT (1 << 10)
  #endif
  #ifndef MAX_IT
  #define MAX_IT (1 << 25)
  #endif
  #ifndef DELTA_IT
  #define DELTA_IT 2
  #endif
  #ifndef REPEATS
  #define REPEATS 10
  #endif
#endif

int main();
std::string getFile(std::string fname);
void onResize(GLFWwindow *, int, int);
void onScroll(GLFWwindow *, double, double);
void renderLoop(glWin &);
void procInput(GLFWwindow *, float, transform &);

#if ENABLE_CAM_ROTATION
void onMouse(GLFWwindow *, double, double);
#endif

#endif
