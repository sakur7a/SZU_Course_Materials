#version 330 core
in vec3 worldPosition;
in vec3 worldNormal;
in vec2 uv;
in vec4 shadowPosition;
flat in float treeVariation;
out vec4 fragmentColor;

uniform sampler2D colorTexture;
uniform sampler2D shadowMap;
uniform vec3 baseColor;
uniform vec2 textureScale;
uniform vec3 lightDirection;
uniform vec3 eyePosition;
uniform float timeSeconds;
uniform int surfaceType;
uniform int shadowsEnabled;
uniform int alphaCutout;

// 光空间深度比较与 3×3 PCF，减轻阴影边缘锯齿。
float shadowAmount(vec3 n, vec3 light) {
    if (shadowsEnabled == 0) return 0.0;
    vec3 p = shadowPosition.xyz / shadowPosition.w;
    p = p * 0.5 + 0.5;
    if (p.z > 1.0 || p.z < 0.0) return 0.0;
    float bias = max(0.0035 * (1.0 - dot(n, light)), 0.0015);
    vec2 texel = 1.0 / vec2(textureSize(shadowMap, 0));
    float shadow = 0.0;
    for (int x = -1; x <= 1; ++x)
        for (int y = -1; y <= 1; ++y)
            shadow += p.z - bias > texture(shadowMap, p.xy + vec2(x, y) * texel).r ? 1.0 : 0.0;
    return shadow / 9.0;
}

void main() {
    vec3 n = normalize(worldNormal);
    vec3 light = normalize(lightDirection);
    vec3 viewDir = normalize(eyePosition - worldPosition);
    vec4 texel = texture(colorTexture, uv * textureScale);
    if ((alphaCutout == 1 && texel.a < 0.12) || (alphaCutout == 2 && texel.a < 0.48)) discard;
    vec3 albedo = texel.rgb * baseColor;
    if (surfaceType == 2) albedo *= mix(vec3(.88,.92,.86),vec3(1.05,1.03,.97),treeVariation);
    if (surfaceType == 1) {
        // 水面为轻量的颜色扰动，不依赖额外的实时反射贴图。
        float ripple = sin(worldPosition.x * .27 + worldPosition.z * .18 + timeSeconds * .35)
                     + sin(worldPosition.x * .09 - worldPosition.z * .33 - timeSeconds * .24);
        albedo *= 0.98 + 0.015 * ripple;
        albedo = mix(albedo, vec3(0.32, 0.45, 0.35), 0.12);
    }
    float diffuse = surfaceType == 2 ? max(abs(dot(n, light)), 0.15) : max(dot(n, light), 0.0);
    vec3 halfVector = normalize(light + viewDir);
    float shininess = surfaceType == 1 ? 72.0 : 32.0;
    float specular = pow(max(dot(n, halfVector), 0.0), shininess);
    float shadow = shadowAmount(n, light);
    if (surfaceType == 2) shadow *= 0.40;
    // 白色厚墙采用较高的环境反射，同时保留窗洞侧壁的方向光和阴影。
    float ambient = (surfaceType == 2 ? 0.60 : surfaceType == 4 ? 0.64 : 0.46) + 0.08 * max(n.y, 0.0);
    vec3 color = albedo * (ambient + 0.55 * diffuse * (1.0 - 0.74 * shadow));
    // 羽毛材质使用低高光，避免将天鹅呈现为反光塑料。
    color += vec3(1.0, 0.96, 0.88) * specular * (surfaceType == 1 ? 0.36 : surfaceType == 5 ? 0.012 : 0.12) * (1.0 - shadow);
    if (surfaceType == 1) {
        // 水面使用方向相关的天空色和低幅波纹，属于近似反射而非真实场景镜像。
        float w1=sin(worldPosition.x*.31 + worldPosition.z*.16 + timeSeconds*.38);
        float w2=sin(worldPosition.x*.10 - worldPosition.z*.22 - timeSeconds*.29);
        vec3 waterNormal=normalize(vec3(.035*w1,1.0,.035*w2));
        float grazing=pow(1.0-max(dot(waterNormal,viewDir),0.0),3.0);
        vec3 skyTint=mix(vec3(.50,.64,.63),vec3(.69,.75,.72),.5+.25*w1+.25*w2);
        color=mix(color,skyTint,.14+.32*grazing);
    }
    float fog = clamp((length(eyePosition - worldPosition) - 1300.0) / 3500.0, 0.0, 0.10);
    color = mix(color, vec3(0.73, 0.81, 0.85), fog);
    // 召南桥护栏以少量透明色模拟玻璃；其他材质保持不透明。
    fragmentColor = vec4(max(color, vec3(0.0)), surfaceType == 3 ? 0.28 : 1.0);
}
