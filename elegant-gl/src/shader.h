#ifndef SHADER_H
#define SHADER_H

#include "glad/include/glad/glad.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>

class Shader
{
    public:
        unsigned int id;

        Shader(const char *, const char *);
        void use();
        void setBool(const std::string &, bool) const;
        void setInt(const std::string &, int) const;
        void setFloat(const std::string &, float) const;
        void setMat4(const std::string &, glm::mat4 *) const;
        void setVec4(const std::string &, glm::vec4 *) const;
};

void printLog(GLuint object);

#endif
