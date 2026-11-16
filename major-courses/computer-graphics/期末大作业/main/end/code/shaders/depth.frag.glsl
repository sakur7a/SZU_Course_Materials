#version 330 core
in vec2 uv;
uniform sampler2D colorTexture;
uniform int alphaCutout;
void main() {
    float alpha = texture(colorTexture, uv).a;
    if ((alphaCutout == 1 && alpha < 0.12) || (alphaCutout == 2 && alpha < 0.48)) discard;
}
