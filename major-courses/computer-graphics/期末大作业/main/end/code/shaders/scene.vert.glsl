#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texCoord;
layout(location = 3) in vec4 instanceData;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat4 lightSpace;
uniform int useInstancing;
out vec3 worldPosition;
out vec3 worldNormal;
out vec2 uv;
out vec4 shadowPosition;
flat out float treeVariation;

void main() {
    mat4 actualModel = model;
    treeVariation = 0.5;
    if (useInstancing == 1) {
        // 每棵树复用相同模型，依据位置生成稳定朝向；w 是树的比例。
        float a = mod(instanceData.x * 1.37 + instanceData.z * .72, 6.2831853);
        float c = cos(a) * instanceData.w, s = sin(a) * instanceData.w;
        mat4 local = mat4(vec4(c,0,-s,0), vec4(0,instanceData.w,0,0),
                          vec4(s,0,c,0), vec4(instanceData.xyz + vec3(0,.18,0),1));
        actualModel = model * local;
        treeVariation = fract(sin(dot(instanceData.xz,vec2(12.9898,78.233))) * 43758.5453);
    }
    vec4 world = actualModel * vec4(position, 1.0);
    worldPosition = world.xyz;
    worldNormal = transpose(inverse(mat3(actualModel))) * normal;
    uv = texCoord;
    shadowPosition = lightSpace * world;
    gl_Position = projection * view * world;
}
