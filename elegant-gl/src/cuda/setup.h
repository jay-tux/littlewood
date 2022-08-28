#ifndef _JAY_CUDA_SETUP
#define _JAY_CUDA_SETUP

#include "glad/include/glad/glad.h"
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include "../shader.h"
//#include "helper_cuda.h"
//#include "helper_gl.h"

typedef unsigned char uchar;

struct cuData {
  uint width, height;
  float tol;
  long maxit;
};

struct cuGL {
  float *buffer;
  float *itbuffer;
  uint gl_tex1, gl_tex2;
};

void createGLTextureForCUDA(cuData &ref, cuGL &bind);
void initCUDABuffers(cuData &ref, cuGL &bind);
void updateShader(Shader &s, const uint tex);
void updateTexture(cuData &ref, cuGL &bind);

#endif
