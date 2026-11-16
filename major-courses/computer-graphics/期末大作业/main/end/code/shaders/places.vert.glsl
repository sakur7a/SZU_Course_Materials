#version 330 core
layout (location=0) in vec3 clipPosition;
layout (location=1) in vec2 texCoord;
layout (location=2) in vec2 anchorUV;
out vec2 uv;
flat out vec3 labelAnchor;
void main() {
    gl_Position=vec4(clipPosition,1.0);
    uv=texCoord;
    labelAnchor=vec3(anchorUV,clipPosition.z*.5+.5);
}
