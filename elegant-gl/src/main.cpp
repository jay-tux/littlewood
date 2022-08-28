#include "main.h"

int main()
{
    glWin win = {
      800, 600, "OpenGL", nullptr, nullptr,
#if ENABLE_CAM_ROTATION
      onMouse,
#endif
      onResize, onScroll
    };

    if(!setup_GLFW_GLAD(win)) return -1;
    renderLoop(win);
    return teardown_GLFW_GLAD(win) ? 0 : -1;
}

// <editor-fold> Callbacks
void onResize(GLFWwindow *win, int w, int h)
{
    glWin::core.cam->scrSizeCam(w, h);
}

void onScroll(GLFWwindow *win, double x, double y)
{
    glWin::core.cam->zoomCam(y);
}

#if ENABLE_CAM_ROTATION
void onMouse(GLFWwindow *win, double x, double y)
{
    glWin::core.cam->rotateCam(x, y);
}
#endif

void procInput(GLFWwindow *win, float delta, transform &t)
{
    if(glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(win, true);

    transformer d = {
      glfwGetKey(win, GLFW_KEY_W) == GLFW_PRESS, glfwGetKey(win, GLFW_KEY_S) == GLFW_PRESS,
      glfwGetKey(win, GLFW_KEY_A) == GLFW_PRESS, glfwGetKey(win, GLFW_KEY_D) == GLFW_PRESS,
      glfwGetKey(win, GLFW_KEY_E) == GLFW_PRESS, glfwGetKey(win, GLFW_KEY_Q) == GLFW_PRESS,
      glfwGetKey(win, GLFW_KEY_SPACE) == GLFW_PRESS
    };

    recenter(d, t);
}
// </editor-fold>

void renderLoop(glWin &win)
{
    fps_t fps = { 0.0f, 0.0f, 0.0f };
    glRender r;
    setup_render(r);
    r.m = glm::translate(r.m, glm::vec3(-1.25f, 0.0f, 0.0f));

    glRender r2;
    setup_render(r2);
    r2.m = glm::translate(r2.m, glm::vec3(1.25f, 0.0f, 0.0f));

    Shader use("shaders/shader.v.glsl", "shaders/shader.f.glsl");
    Shader sec("shaders/shader.v.glsl", "shaders/iter.f.glsl");

    cuData ref  = { 1024, 1024, 1e-20, (1 << 20) };
    cuGL   bind = { nullptr, 0 };
    transform t = { { 0.0f, 0.0f }, 1.0f, true };

    //findCudaDevice(0, nullptr);
    GL_CHECK(initCUDABuffers(ref, bind));
    GL_CHECK(createGLTextureForCUDA(ref, bind));

    auto addtex1 = [bind](auto sh) {
      updateShader(sh, bind.gl_tex1);
    };

    auto addtex2 = [bind](auto sh) {
      updateShader(sh, bind.gl_tex2);
    };
#if PROFILER
    for(float prc = MIN_PRC; prc <= MAX_PRC; prc *= DELTA_IT) {
      for(long maxit = MIN_IT; maxit <= MAX_IT; maxit *= DELTA_IT) {
        for(int i = 0; i < REPEATS; i++) {
          ref = { 1024, 1024, prc, maxit };
          glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
          std::cout << prc << "," << maxit << ",";
          GL_CHECK(calcfps(fps));
          std::cout << std::endl;
          GL_CHECK(callKernel(ref, bind, t));
          GL_CHECK(update_render(r,  use, win, addtex1));
          GL_CHECK(update_render(r2, sec, win, addtex2));

          glfwSwapBuffers(win.win);
        }
      }
    }
#else
    while(!glfwWindowShouldClose(win.win))
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        GL_CHECK(calcfps(fps));
        GL_CHECK(procInput(win.win, fps.delta, t));
        if(t.hasChanged) {
          std::cout << "\nCalling kernel, transform: " << t.center.x << "x" << t.center.y << ";" << t.zoom << std::endl;
          glfwSwapBuffers(win.win);
          GL_CHECK(callKernel(ref, bind, t));
          t.hasChanged = false;
        }
        GL_CHECK(update_render(r,  use, win, addtex1));
        GL_CHECK(update_render(r2, sec, win, addtex2));

        glfwSwapBuffers(win.win);
        glfwPollEvents();
    }
#endif

    teardown_render(r);
}
