#version 330 core
in vec2 uv;
out vec4 fragmentColor;
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1,311.7))) * 43758.5453); }
float noise(vec2 p) {
    vec2 i = floor(p), f = fract(p); f = f*f*(3.0-2.0*f);
    return mix(mix(hash(i),hash(i+vec2(1,0)),f.x),mix(hash(i+vec2(0,1)),hash(i+vec2(1,1)),f.x),f.y);
}
void main() {
    float h = clamp(uv.y, 0.0, 1.0);
    vec3 horizon = vec3(0.78, 0.87, 0.90);
    vec3 zenith = vec3(0.31, 0.56, 0.76);
    vec3 sky = mix(horizon, zenith, pow(h, 0.74));
    vec2 cloudPosition = uv * vec2(4.2, 7.5) + vec2(1.3, 2.7);
    float cloud = noise(cloudPosition) * 0.58 + noise(cloudPosition * 2.1) * 0.28 + noise(cloudPosition * 4.2) * 0.14;
    cloud = smoothstep(0.59, 0.76, cloud) * smoothstep(0.20, 0.65, h);
    sky = mix(sky, vec3(0.92,0.94,0.93), cloud * 0.44);
    float sun = pow(max(0.0, 1.0 - length((uv - vec2(0.78, 0.78)) * vec2(1.0, 1.55))), 30.0);
    sky += vec3(0.14, 0.12, 0.08) * sun;
    fragmentColor = vec4(sky, 1.0);
}
