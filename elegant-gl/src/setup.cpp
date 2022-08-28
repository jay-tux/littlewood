#include "setup.h"

glWin glWin::core;

bool setup_GLFW_GLAD(glWin &win) {
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  win.win = glfwCreateWindow(win.w, win.h, win.title.c_str(), nullptr, nullptr);
  if(win.win == nullptr)
  {
      std::cout << "Can't open GLFW window" << std::endl;
      glfwTerminate();
      return false;
  }
  glfwMakeContextCurrent(win.win);

  if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
  {
      std::cout << "Can't start GLAD" << std::endl;
      glfwTerminate();
      return false;
  }

  glViewport(0, 0, win.w, win.h);
  glEnable(GL_DEPTH_TEST);
#if ENABLE_CAM_ROTATION
  glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
  glfwSetCursorPosCallback(win.win, win.onMouse); //enable cam rotation on mouse
#endif
  glfwSetScrollCallback(win.win, win.onScroll);
  glfwSetFramebufferSizeCallback(win.win, win.onResize);

  win.cam = new Camera();
  win.cam->scrSizeCam(win.w, win.h);

  glWin::core = win;
  return true;
}

bool teardown_GLFW_GLAD(glWin &win) {
  delete win.cam;
  glfwTerminate();
  return true;
}

void setup_render(glRender &render) {
#if !PROFILER
  printf("OpenGL version is (%s)\n", glGetString(GL_VERSION));
#endif
  float x = 1.0f;
  float vertices[] = {
      //vertices    //texcoords
      -x, -x, -x,   0.0f, 0.0f,
       x, -x, -x,   1.0f, 0.0f,
       x,  x, -x,   1.0f, 1.0f,
       x,  x, -x,   1.0f, 1.0f,
      -x,  x, -x,   0.0f, 1.0f,
      -x, -x, -x,   0.0f, 0.0f
  };
  unsigned int indices[] = {
      0, 1, 2,
      0, 2, 3
  };
  glGenBuffers(1, &render.vbo);
  glGenBuffers(1, &render.ebo);
  glGenVertexArrays(1, &render.vao);

  glBindVertexArray(render.vao);
  glBindBuffer(GL_ARRAY_BUFFER, render.vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)(0));
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)(3 * sizeof(float)));
  glEnableVertexAttribArray(1);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, render.ebo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

  //unbind -> prevent accidental modifications
  glBindVertexArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

  glClearColor(0.0f, 0.0f, 0.15f, 1.0f);
  render.m = glm::mat4(1.0f);
}

void update_render(glRender &render, Shader &use, glWin &win, func mod) {
  win.cam->update(&render.v, &render.p);

  use.use();
  use.setMat4("model",      &render.m);
  use.setMat4("view",       &render.v);
  use.setMat4("projection", &render.p);
  mod(use);
  glBindVertexArray(render.vao);
  glDrawArrays(GL_TRIANGLES, 0, 18);
  glBindVertexArray(0);
}

void teardown_render(glRender &render) {
  glDeleteBuffers(1, &render.vbo);
  glDeleteBuffers(1, &render.ebo);
}

void calcfps(fps_t &fps) {
  fps.current = glfwGetTime();
  fps.delta = fps.current - fps.last;
#if PROFILER
  std::cout << 1.0f / fps.delta;
#else
  std::cout << '\r' << 1.0f / fps.delta << " fps    " << std::flush;
#endif
  fps.last = fps.current;
}
