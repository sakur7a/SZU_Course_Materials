#pragma once

#include <glad/glad.h>

// 从 UTF-8 文本文件编译并链接一组顶点、片元着色器。
GLuint loadShaderProgram(const char* vertexPath, const char* fragmentPath);
