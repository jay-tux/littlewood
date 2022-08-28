#ifndef _JAY_GL_SETUP
#define _JAY_GL_SETUP

#include "glad/include/glad/glad.h"
#include <GLFW/glfw3.h>
#include <iostream>
#include <functional>
#include "camera.h"
#include "shader.h"

#ifndef ENABLE_CAM_ROTATION
#define ENABLE_CAM_ROTATION 0
#endif

#ifndef PROFILER
#define PROFILER 0
#endif

typedef std::function<void(Shader &)> func;

struct glWin {
  uint w, h;
  std::string title;
  GLFWwindow *win;
  Camera *cam;
#if ENABLE_CAM_ROTATION
  void (*onMouse)(GLFWwindow *, double, double);
#endif
  void (*onResize)(GLFWwindow *, int, int);
  void (*onScroll)(GLFWwindow *, double, double);

  static glWin core;
};

bool setup_GLFW_GLAD(glWin &win);
bool teardown_GLFW_GLAD(glWin &win);

struct glRender {
  uint vbo, vao, ebo;
  glm::mat4 m, v, p;
};

void setup_render(glRender &render);
void update_render(glRender &render, Shader &use, glWin &win, func mod);
void teardown_render(glRender & render);

struct fps_t {
  float last, delta, current;
};

void calcfps(fps_t &fps);

#endif
