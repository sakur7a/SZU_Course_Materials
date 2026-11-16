#version 330 core
in vec2 uv;
out vec4 FragColor;
uniform sampler2D labelAtlas;
uniform sampler2D sceneDepth;
flat in vec3 labelAnchor;
void main() {
    if (texture(sceneDepth,labelAnchor.xy).r+0.000002<labelAnchor.z) discard;
    vec4 color=texture(labelAtlas,uv);
    if (color.a<0.02) discard;
    FragColor=color;
}
