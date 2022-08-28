#ifndef _JAY_GL_DEBUG
#define _JAY_GL_DEBUG

#include "glad/include/glad/glad.h"
#include <GLFW/glfw3.h>
#include <iostream>

void CheckOpenGLError(const char* stmt, const char* fname, int line);

#define GL_CHECK(stmt) do { \
        stmt; \
        CheckOpenGLError(#stmt, __FILE__, __LINE__); \
    } while (0)

#endif
