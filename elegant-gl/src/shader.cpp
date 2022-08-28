#include "shader.h"

void printLog(GLuint object) {
    GLint log_length = 0;
    if (glIsShader(object)) {
        glGetShaderiv(object, GL_INFO_LOG_LENGTH, &log_length);
    } else if (glIsProgram(object)) {
        glGetProgramiv(object, GL_INFO_LOG_LENGTH, &log_length);
    } else {
        std::cerr << "printlog: Not a shader or a program" << std::endl;
        return;
    }

    char* log = (char*)calloc(log_length, sizeof(char));

    if (glIsShader(object))
        { glGetShaderInfoLog(object, log_length, NULL, log); }
    else if (glIsProgram(object))
        { glGetProgramInfoLog(object, log_length, NULL, log); }

    std::cerr << log;
    free(log);
}

Shader::Shader(const char *vpath, const char *fpath)
{
    std::string vcode, fcode;
    std::ifstream vfile, ffile;
    vfile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    ffile.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    try
    {
        vfile.open(vpath); ffile.open(fpath);
        std::stringstream vstr, fstr;
        vstr << vfile.rdbuf(); fstr << ffile.rdbuf();
        vfile.close(); ffile.close();
        vcode = vstr.str(); fcode = fstr.str();
    }
    catch(std::ifstream::failure &e)
    {
        std::cerr << "Error reading shader files:" << std::endl <<
            std::strerror(errno) << std::endl;
    }

    const char *vscode = vcode.c_str();
    const char *fscode = fcode.c_str();

    unsigned int v, f;
    int success;
    v = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(v, 1, &vscode, nullptr);
    glCompileShader(v);
    glGetShaderiv(v, GL_COMPILE_STATUS, &success);
    if(!success)
    {
        std::cerr << "Failed compiling " << vpath << ":" << std::endl;
        printLog(v);
    }

    f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(f, 1, &fscode, nullptr);
    glCompileShader(f);
    glGetShaderiv(f, GL_COMPILE_STATUS, &success);
    if(!success)
    {
        std::cerr << "Failed compiling " << fpath << ":" << std::endl;
        printLog(f);
    }

    id = glCreateProgram();
    glAttachShader(id, v); glAttachShader(id, f);
    glLinkProgram(id);
    glGetProgramiv(id, GL_LINK_STATUS, &success);
    if(!success)
    {
        std::cerr << "Failed linking " << vpath << " and " << fpath << ":" << std::endl;
        printLog(id);
    }

    glDeleteShader(v); glDeleteShader(f);
}

void Shader::use() { glUseProgram(id); }

void Shader::setBool(const std::string &name, bool value) const
{ glUniform1i(glGetUniformLocation(id, name.c_str()), (int)value); }

void Shader::setInt(const std::string &name, int value) const
{ glUniform1i(glGetUniformLocation(id, name.c_str()), value); }

void Shader::setFloat(const std::string &name, float value) const
{ glUniform1f(glGetUniformLocation(id, name.c_str()), value); }

void Shader::setMat4(const std::string &name, glm::mat4 *mat) const
{
    auto matptr = glm::value_ptr(*mat);
    glUniformMatrix4fv(glGetUniformLocation(id, name.c_str()), 1, GL_FALSE, matptr);
}

void Shader::setVec4(const std::string &name, glm::vec4 *vec) const
{
    auto vecptr = glm::value_ptr(*vec);
    glUniform4fv(glGetUniformLocation(id, name.c_str()), 1, vecptr);
}
