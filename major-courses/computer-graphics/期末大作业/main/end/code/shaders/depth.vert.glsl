#version 330 core
layout(location = 0) in vec3 position;
layout(location = 2) in vec2 texCoord;
layout(location = 3) in vec4 instanceData;
uniform mat4 model;
uniform mat4 lightSpace;
uniform int useInstancing;
out vec2 uv;
void main() {
    mat4 actualModel = model;
    if (useInstancing == 1) {
        float a = mod(instanceData.x * 1.37 + instanceData.z * .72, 6.2831853);
        float c = cos(a) * instanceData.w, s = sin(a) * instanceData.w;
        mat4 local = mat4(vec4(c,0,-s,0), vec4(0,instanceData.w,0,0),
                          vec4(s,0,c,0), vec4(instanceData.xyz + vec3(0,.18,0),1));
        actualModel = model * local;
    }
    uv = texCoord;
    gl_Position = lightSpace * actualModel * vec4(position, 1.0);
}
