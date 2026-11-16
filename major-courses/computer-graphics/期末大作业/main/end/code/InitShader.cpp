#include "Shader.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// 把着色器文件读入内存；文件缺失时返回清晰的错误，而不是继续绘制黑屏。
static std::string readText(const char* path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error(std::string("Cannot open shader: ") + path);
    std::ostringstream content;
    content << file.rdbuf();
    return content.str();
}

static GLuint compile(GLenum type, const char* path) {
    std::string text = readText(path);
    const char* source = text.c_str();
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    GLint ok = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLint size = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &size);
        std::vector<char> log(static_cast<size_t>(size > 1 ? size : 1), 0);
        glGetShaderInfoLog(shader, size, NULL, &log[0]);
        std::string message = std::string(path) + ": " + &log[0];
        glDeleteShader(shader);
        throw std::runtime_error(message);
    }
    return shader;
}

GLuint loadShaderProgram(const char* vertexPath, const char* fragmentPath) {
    GLuint vertex = compile(GL_VERTEX_SHADER, vertexPath);
    GLuint fragment = compile(GL_FRAGMENT_SHADER, fragmentPath);
    GLuint program = glCreateProgram();
    glAttachShader(program, vertex);
    glAttachShader(program, fragment);
    glLinkProgram(program);
    glDeleteShader(vertex);
    glDeleteShader(fragment);
    GLint ok = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (!ok) {
        GLint size = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &size);
        std::vector<char> log(static_cast<size_t>(size > 1 ? size : 1), 0);
        glGetProgramInfoLog(program, size, NULL, &log[0]);
        std::string message = std::string("Shader link failed: ") + &log[0];
        glDeleteProgram(program);
        throw std::runtime_error(message);
    }
    return program;
}
