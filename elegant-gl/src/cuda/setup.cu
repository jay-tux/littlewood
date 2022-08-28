#include "setup.h"
#include "gldebug.h"

void createGLTextureForCUDA(cuData &ref, cuGL &bind)
{
	// create an OpenGL texture
	GL_CHECK(glActiveTexture(GL_TEXTURE0));
	GL_CHECK(glGenTextures(1, &bind.gl_tex1));
	GL_CHECK(glBindTexture(GL_TEXTURE_2D, bind.gl_tex1));
	// set basic texture parameters
	GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
	GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
	GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
	GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
	// Specify 2D texture
	GL_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ref.width, ref.height, 0, GL_RED, GL_FLOAT, bind.buffer));

	// create an OpenGL texture
	GL_CHECK(glActiveTexture(GL_TEXTURE0));
	GL_CHECK(glGenTextures(1, &bind.gl_tex2));
	GL_CHECK(glBindTexture(GL_TEXTURE_2D, bind.gl_tex2));
	// set basic texture parameters
	GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
	GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
	GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
	GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
	// Specify 2D texture
	GL_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ref.width, ref.height, 0, GL_RED, GL_FLOAT, bind.itbuffer));
}

void initCUDABuffers(cuData &ref, cuGL &bind)
{
	cudaMallocManaged(&bind.buffer, ref.width * ref.height * sizeof(float));
	for(int i = 0; i < ref.width * ref.height; i++) { bind.buffer[i] = 0.0f; }
	cudaMallocManaged(&bind.itbuffer, ref.width * ref.height * sizeof(float));
	for(int i = 0; i < ref.width * ref.height; i++) { bind.itbuffer[i] = 0.0f; }
}

void updateShader(Shader &s, const uint tex) {
  glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, tex);
  s.setInt("tex", 0);
}

void updateTexture(cuData &ref, cuGL &bind) {
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, bind.gl_tex1);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ref.width, ref.height, 0, GL_RED, GL_FLOAT, bind.buffer);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, bind.gl_tex2);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ref.width, ref.height, 0, GL_RED, GL_FLOAT, bind.itbuffer);
	//glBindTexture(GL_TEXTURE_2D, 0);
}
