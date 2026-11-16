#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Vertex { float p[3], n[3], uv[2]; };
static_assert(sizeof(Vertex) == 8 * sizeof(float), "Unexpected vertex layout");

struct Node {
    std::string id, parent;
    int parentIndex;
    glm::mat4 local;
};
struct Texture {
    unsigned width, height;
    std::vector<unsigned char> rgba;
};
struct Material {
    std::string id, texture;
    bool mask, doubleSided;
    float cutoff;
    glm::vec4 color;
    Texture image;
};
struct Primitive {
    std::string id, node, material, mesh;
    int nodeIndex, materialIndex;
    std::vector<Vertex> vertices;
};
struct Anchor {
    std::string id, node;
    int nodeIndex;
    glm::vec3 local, expected;
};
struct View { glm::vec3 eye, target; };
struct Asset {
    std::vector<Node> nodes;
    std::vector<Material> materials;
    std::vector<Primitive> primitives;
    std::vector<Anchor> anchors;
    std::map<std::string, int> nodeIndex, materialIndex;
    std::map<std::string, View> views;
    std::string interaction, probeParent, probeDescendant;
    int interactionIndex, probeParentIndex, probeDescendantIndex;
    glm::vec3 axis;
    float minAngle, maxAngle, probeAngle;
};
struct Check {
    std::vector<glm::mat4> base, posed;
    glm::vec3 min, max;
    std::vector<std::string> chain;
    glm::mat4 before, after;
    float maxDelta, maxAnchorError;
    size_t triangles;
};

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

// 包内引用只能使用相对路径，不允许通过盘符或上级目录逃离资产目录。
std::string packagePath(const std::string& root, const std::string& relative) {
    require(!relative.empty() && relative[0] != '/' && relative[0] != '\\' &&
            relative.find(':') == std::string::npos, "Invalid package-relative path: " + relative);
    std::string part;
    for (size_t i = 0; i <= relative.size(); ++i) {
        if (i == relative.size() || relative[i] == '/' || relative[i] == '\\') {
            require(!part.empty() && part != "." && part != "..", "Invalid package-relative path: " + relative);
            part.clear();
        } else part.push_back(relative[i]);
    }
    return root + "/" + relative;
}

std::vector<std::string> split(const std::string& line) {
    std::vector<std::string> result;
    size_t start = 0;
    for (;;) {
        size_t at = line.find('\t', start);
        result.push_back(line.substr(start, at == std::string::npos ? at : at - start));
        if (at == std::string::npos) break;
        start = at + 1;
    }
    return result;
}
typedef std::vector<std::vector<std::string> > Rows;
Rows readTsv(const std::string& root, const std::string& name) {
    const std::string path = packagePath(root, name);
    std::ifstream f(path.c_str(), std::ios::binary);
    require(bool(f), "Cannot open TSV: " + path);
    Rows rows;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line[line.size() - 1] == '\r') line.resize(line.size() - 1);
        if (rows.empty() && line.size() >= 3 && static_cast<unsigned char>(line[0]) == 0xef &&
            static_cast<unsigned char>(line[1]) == 0xbb && static_cast<unsigned char>(line[2]) == 0xbf)
            line.erase(0, 3);
        if (line.empty() || line[0] == '#') continue;
        rows.push_back(split(line));
    }
    require(!f.bad(), "Error reading TSV: " + path);
    return rows;
}
float number(const std::string& text, const std::string& where) {
    require(!text.empty(), "Empty number in " + where);
    char* end = NULL;
    const double x = std::strtod(text.c_str(), &end);
    require(end != text.c_str() && *end == '\0' && std::isfinite(x) &&
            std::fabs(x) <= std::numeric_limits<float>::max(), "Invalid number in " + where + ": " + text);
    return static_cast<float>(x);
}
void fields(const std::vector<std::string>& row, size_t count, const std::string& where) {
    require(row.size() == count, "Wrong column count in " + where);
}
glm::vec3 vec3(const std::vector<std::string>& r, size_t offset, const std::string& where) {
    return glm::vec3(number(r[offset], where), number(r[offset+1], where), number(r[offset+2], where));
}
int lookup(const std::map<std::string, int>& index, const std::string& id, const std::string& what) {
    std::map<std::string, int>::const_iterator it = index.find(id);
    require(it != index.end(), "Unknown " + what + ": " + id);
    return it->second;
}
void addId(std::map<std::string, int>& index, const std::string& id, int value, const std::string& kind) {
    require(!id.empty() && id != "-", "Invalid " + kind + " ID");
    require(index.insert(std::make_pair(id, value)).second, "Duplicate " + kind + " ID: " + id);
}

std::vector<unsigned char> bytes(const std::string& path) {
    std::ifstream f(path.c_str(), std::ios::binary | std::ios::ate);
    require(bool(f), "Cannot open resource: " + path);
    const std::streamoff n = f.tellg();
    require(n >= 0 && n <= static_cast<std::streamoff>(1024) * 1024 * 1024,
            "Invalid or excessively large resource: " + path);
    std::vector<unsigned char> result(static_cast<size_t>(n));
    f.seekg(0);
    if (!result.empty()) f.read(reinterpret_cast<char*>(&result[0]), static_cast<std::streamsize>(result.size()));
    require(bool(f), "Cannot read resource: " + path);
    return result;
}
std::uint32_t uint32le(const unsigned char* p) {
    return std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8) |
           (std::uint32_t(p[2]) << 16) | (std::uint32_t(p[3]) << 24);
}
float float32le(const unsigned char* p) {
    const std::uint32_t bits = uint32le(p);
    float value;
    static_assert(sizeof(value) == sizeof(bits), "SZM1 requires 32-bit float");
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

// SZM1 以节点局部坐标展开三角形；文件大小与每个浮点值都要验证。
std::vector<Vertex> readMesh(const std::string& path) {
    const std::vector<unsigned char> data = bytes(path);
    require(data.size() >= 8 && std::memcmp(&data[0], "SZM1", 4) == 0, "Invalid SZM1 header: " + path);
    const std::uint32_t count = uint32le(&data[4]);
    require(count && count % 3 == 0 && count <= 10000000 &&
            data.size() == 8 + static_cast<size_t>(count) * 32, "Invalid SZM1 size or vertex count: " + path);
    std::vector<Vertex> out(count);
    for (size_t i = 0; i < count; ++i) {
        float values[8];
        for (size_t k = 0; k < 8; ++k) {
            values[k] = float32le(&data[8 + i * 32 + k * 4]);
            require(std::isfinite(values[k]), "Non-finite SZM1 vertex: " + path);
        }
        require(values[3]*values[3]+values[4]*values[4]+values[5]*values[5] > 1e-12f,
                "Zero SZM1 normal: " + path);
        std::memcpy(&out[i], values, sizeof(values));
    }
    return out;
}
Texture readTexture(const std::string& path) {
    const std::vector<unsigned char> data = bytes(path);
    require(data.size() >= 8, "Invalid RGBA texture header: " + path);
    Texture t; t.width = uint32le(&data[0]); t.height = uint32le(&data[4]);
    require(t.width && t.height && t.width <= 8192 && t.height <= 8192 &&
            data.size() == 8 + static_cast<size_t>(t.width) * t.height * 4, "Invalid RGBA texture size: " + path);
    t.rgba.assign(data.begin() + 8, data.end());
    return t;
}

Asset loadAsset(const std::string& root) {
    Asset a;
    a.interactionIndex = a.probeParentIndex = a.probeDescendantIndex = -1;
    Rows rows = readTsv(root, "legacy/nodes.tsv");
    for (size_t i = 0; i < rows.size(); ++i) {
        const std::vector<std::string>& r = rows[i]; fields(r, 20, "nodes.tsv");
        require(r[0] == "N", "Unknown nodes.tsv record");
        Node n; n.id = r[1]; n.parent = r[2]; n.parentIndex = -1;
        addId(a.nodeIndex, n.id, static_cast<int>(a.nodes.size()), "node");
        for (int c = 0; c < 4; ++c) for (int y = 0; y < 4; ++y)
            n.local[c][y] = number(r[4 + c*4 + y], "nodes.tsv");
        require(std::fabs(n.local[0][3]) < 1e-6f && std::fabs(n.local[1][3]) < 1e-6f &&
                std::fabs(n.local[2][3]) < 1e-6f && std::fabs(n.local[3][3] - 1) < 1e-6f,
                "Non-affine local matrix: " + n.id);
        require(std::fabs(glm::determinant(glm::mat3(n.local))) > 1e-10f, "Singular local matrix: " + n.id);
        a.nodes.push_back(n);
    }
    require(!a.nodes.empty(), "No nodes");
    for (size_t i = 0; i < a.nodes.size(); ++i)
        if (a.nodes[i].parent != "-") a.nodes[i].parentIndex = lookup(a.nodeIndex, a.nodes[i].parent, "parent node");

    rows = readTsv(root, "legacy/materials.tsv");
    for (size_t i = 0; i < rows.size(); ++i) {
        const std::vector<std::string>& r = rows[i]; fields(r, 10, "materials.tsv");
        require(r[0] == "T", "Unknown materials.tsv record");
        Material m; m.id = r[1]; m.texture = r[2];
        addId(a.materialIndex, m.id, static_cast<int>(a.materials.size()), "material");
        require(r[3] == "OPAQUE" || r[3] == "MASK", "Unsupported alpha mode: " + r[3]);
        m.mask = r[3] == "MASK"; m.cutoff = number(r[4], "materials.tsv");
        require(m.cutoff >= 0 && m.cutoff <= 1, "Invalid alpha cutoff: " + m.id);
        require(r[5] == "0" || r[5] == "1", "Invalid double-sided flag: " + m.id);
        m.doubleSided = r[5] == "1";
        for (int k = 0; k < 4; ++k) {
            m.color[k] = number(r[6+k], "materials.tsv");
            require(m.color[k] >= 0 && m.color[k] <= 1, "Invalid base color: " + m.id);
        }
        if (m.texture != "-") m.image = readTexture(packagePath(root, m.texture));
        a.materials.push_back(m);
    }
    require(!a.materials.empty(), "No materials");

    std::map<std::string, int> primitiveIds, anchorIds;
    rows = readTsv(root, "legacy/primitives.tsv");
    for (size_t i = 0; i < rows.size(); ++i) {
        const std::vector<std::string>& r = rows[i]; fields(r, 5, "primitives.tsv");
        require(r[0] == "P", "Unknown primitives.tsv record");
        Primitive p; p.id = r[1]; p.node = r[2]; p.material = r[3]; p.mesh = r[4];
        addId(primitiveIds, p.id, static_cast<int>(a.primitives.size()), "primitive");
        p.nodeIndex = lookup(a.nodeIndex, p.node, "primitive node");
        p.materialIndex = lookup(a.materialIndex, p.material, "primitive material");
        p.vertices = readMesh(packagePath(root, p.mesh));
        a.primitives.push_back(p);
    }
    require(!a.primitives.empty(), "No primitives");
    rows = readTsv(root, "legacy/anchors.tsv");
    for (size_t i = 0; i < rows.size(); ++i) {
        const std::vector<std::string>& r = rows[i]; fields(r, 9, "anchors.tsv");
        require(r[0] == "A", "Unknown anchors.tsv record");
        Anchor b; b.id = r[1]; b.node = r[2];
        addId(anchorIds, b.id, static_cast<int>(a.anchors.size()), "anchor");
        b.nodeIndex = lookup(a.nodeIndex, b.node, "anchor node");
        b.local = vec3(r, 3, "anchors.tsv"); b.expected = vec3(r, 6, "anchors.tsv");
        a.anchors.push_back(b);
    }
    require(a.anchors.size() >= 3, "At least three anchors are required");

    bool gotI = false, gotH = false;
    rows = readTsv(root, "legacy/controls.tsv");
    for (size_t i = 0; i < rows.size(); ++i) {
        const std::vector<std::string>& r = rows[i];
        if (r[0] == "I") {
            fields(r, 7, "controls.tsv I"); require(!gotI, "Duplicate interaction"); gotI = true;
            a.interaction = r[1]; a.interactionIndex = lookup(a.nodeIndex, r[1], "interaction node");
            a.axis = vec3(r, 2, "controls.tsv I");
            require(glm::length(a.axis) > 1e-6f, "Zero interaction axis"); a.axis = glm::normalize(a.axis);
            a.minAngle = number(r[5], "controls.tsv I"); a.maxAngle = number(r[6], "controls.tsv I");
            require(a.minAngle <= a.maxAngle, "Invalid interaction range");
        } else if (r[0] == "H") {
            fields(r, 4, "controls.tsv H"); require(!gotH, "Duplicate hierarchy probe"); gotH = true;
            a.probeParent = r[1]; a.probeDescendant = r[2];
            a.probeParentIndex = lookup(a.nodeIndex, r[1], "probe parent");
            a.probeDescendantIndex = lookup(a.nodeIndex, r[2], "probe descendant");
            a.probeAngle = number(r[3], "controls.tsv H");
            require(std::fabs(a.probeAngle) > 1e-5f, "Zero hierarchy probe rotation");
        } else if (r[0] == "V") {
            fields(r, 8, "controls.tsv V");
            View v; v.eye = vec3(r, 2, "controls.tsv V"); v.target = vec3(r, 5, "controls.tsv V");
            require(glm::length(v.eye - v.target) > 1e-5f, "Invalid view: " + r[1]);
            require(a.views.insert(std::make_pair(r[1], v)).second, "Duplicate view: " + r[1]);
        } else throw std::runtime_error("Unknown controls.tsv record: " + r[0]);
    }
    require(gotI && gotH, "Missing interaction or hierarchy probe");
    require(a.views.count("hero") && a.views.count("front") && a.views.count("top"), "Missing required views");
    return a;
}

// 递归计算真实父子关系，检测环，并在节点局部矩阵之后附加动件旋转。
std::vector<glm::mat4> worldMatrices(const Asset& a, int rotateNode, const glm::vec3& axis, float angle) {
    std::vector<glm::mat4> world(a.nodes.size());
    std::vector<int> state(a.nodes.size(), 0);
    std::function<void(int)> visit = [&](int i) {
        require(state[i] != 1, "Node hierarchy contains a cycle at: " + a.nodes[i].id);
        if (state[i] == 2) return;
        state[i] = 1;
        glm::mat4 local = a.nodes[i].local;
        if (i == rotateNode) local = local * glm::rotate(glm::mat4(1), glm::radians(angle), axis);
        if (a.nodes[i].parentIndex >= 0) {
            visit(a.nodes[i].parentIndex);
            world[i] = world[a.nodes[i].parentIndex] * local;
        } else world[i] = local;
        require(std::fabs(glm::determinant(glm::mat3(world[i]))) > 1e-10f,
                "Singular world matrix: " + a.nodes[i].id);
        state[i] = 2;
    };
    for (size_t i = 0; i < a.nodes.size(); ++i) visit(static_cast<int>(i));
    return world;
}
glm::vec3 point(const glm::mat4& m, const glm::vec3& p) { return glm::vec3(m * glm::vec4(p, 1)); }

Check checkAsset(const Asset& a, float angle) {
    require(angle >= a.minAngle - 1e-5f && angle <= a.maxAngle + 1e-5f, "Angle is outside the interaction range");
    Check c;
    c.base = worldMatrices(a, -1, glm::vec3(0,1,0), 0);
    c.posed = worldMatrices(a, a.interactionIndex, a.axis, angle);
    c.maxAnchorError = 0;
    std::vector<glm::vec3> anchorWorld;
    for (size_t i = 0; i < a.anchors.size(); ++i) {
        const Anchor& b = a.anchors[i];
        glm::vec3 actual = point(c.base[b.nodeIndex], b.local);
        float error = glm::length(actual - b.expected);
        c.maxAnchorError = std::max(c.maxAnchorError, error);
        require(error <= 1e-4f, "Anchor world-coordinate mismatch: " + b.id);
        anchorWorld.push_back(actual);
    }
    bool noncollinear = false;
    for (size_t i = 1; i < anchorWorld.size() && !noncollinear; ++i)
        for (size_t j = i+1; j < anchorWorld.size(); ++j)
            if (glm::length(glm::cross(anchorWorld[i]-anchorWorld[0], anchorWorld[j]-anchorWorld[0])) > 1e-5f)
                noncollinear = true;
    require(noncollinear, "Anchor positions are collinear");
    int cursor = a.probeDescendantIndex;
    bool foundParent = false, foundGeometry = false, foundInteractionGeometry = false;
    while (cursor >= 0) {
        c.chain.push_back(a.nodes[cursor].id);
        if (cursor == a.probeParentIndex) foundParent = true;
        cursor = a.nodes[cursor].parentIndex;
    }
    require(foundParent && a.probeParentIndex != a.probeDescendantIndex, "Probe parent is not a strict ancestor");
    std::reverse(c.chain.begin(), c.chain.end());
    require(c.chain.size() >= 4, "Hierarchy does not have a four-node chain to the probe descendant");
    for (size_t i = 0; i < a.primitives.size(); ++i) {
        int p = a.primitives[i].nodeIndex;
        while (p >= 0) {
            if (p == a.probeDescendantIndex) foundGeometry = true;
            if (p == a.interactionIndex) foundInteractionGeometry = true;
            p = a.nodes[p].parentIndex;
        }
    }
    require(foundGeometry, "Hierarchy probe descendant has no actual geometry");
    require(foundInteractionGeometry, "Interaction node has no actual geometry");
    c.before = c.base[a.probeDescendantIndex];
    c.after = worldMatrices(a, a.probeParentIndex, glm::vec3(0,1,0), a.probeAngle)[a.probeDescendantIndex];
    c.maxDelta = 0;
    for (int x = 0; x < 4; ++x) for (int y = 0; y < 4; ++y)
        c.maxDelta = std::max(c.maxDelta, std::fabs(c.after[x][y] - c.before[x][y]));
    require(c.maxDelta > 1e-5f, "Rotating probe parent did not change descendant world transform");

    c.min = glm::vec3(std::numeric_limits<float>::max());
    c.max = glm::vec3(-std::numeric_limits<float>::max());
    c.triangles = 0;
    for (size_t i = 0; i < a.primitives.size(); ++i) {
        const Primitive& p = a.primitives[i]; c.triangles += p.vertices.size() / 3;
        for (size_t j = 0; j < p.vertices.size(); ++j) {
            const Vertex& v = p.vertices[j];
            const glm::vec3 q = point(c.posed[p.nodeIndex], glm::vec3(v.p[0],v.p[1],v.p[2]));
            require(std::isfinite(q.x) && std::isfinite(q.y) && std::isfinite(q.z), "Invalid world-space vertex");
            c.min = glm::min(c.min, q); c.max = glm::max(c.max, q);
        }
    }
    require(c.max.x > c.min.x && c.max.y > c.min.y && c.max.z > c.min.z, "Degenerate world bounds");
    return c;
}

std::string jsonString(const std::string& s) {
    std::ostringstream out; out << '"';
    for (size_t i = 0; i < s.size(); ++i) {
        unsigned char ch = static_cast<unsigned char>(s[i]);
        if (ch == '"' || ch == '\\') out << '\\' << char(ch);
        else if (ch < 0x20) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(ch) << std::dec;
        else out << char(ch);
    }
    out << '"'; return out.str();
}
void jsonVec(std::ostream& out, const glm::vec3& v) { out << '[' << v.x << ',' << v.y << ',' << v.z << ']'; }
void jsonMat(std::ostream& out, const glm::mat4& m) {
    out << '['; for (int i = 0; i < 16; ++i) { if (i) out << ','; out << m[i/4][i%4]; } out << ']';
}
std::string report(const Asset& a, const Check& c, float angle) {
    std::ostringstream o; o << std::setprecision(9);
    o << "{\n  \"status\":\"pass\",\n  \"angle_deg\":" << angle << ",\n  \"bounds\":{\"min\":";
    jsonVec(o, c.min); o << ",\"max\":"; jsonVec(o, c.max); o << "},\n  \"anchors\":[";
    for (size_t i = 0; i < a.anchors.size(); ++i) {
        const Anchor& b = a.anchors[i]; if (i) o << ',';
        glm::vec3 actual = point(c.base[b.nodeIndex], b.local);
        o << "{\"id\":" << jsonString(b.id) << ",\"world\":"; jsonVec(o, actual);
        o << ",\"expected_world\":"; jsonVec(o, b.expected);
        o << ",\"posed_world\":"; jsonVec(o, point(c.posed[b.nodeIndex], b.local));
        o << ",\"error\":" << glm::length(actual-b.expected) << '}';
    }
    o << "],\n  \"max_anchor_error\":" << c.maxAnchorError << ",\n  \"hierarchy\":{\"parent\":"
      << jsonString(a.probeParent) << ",\"descendant\":" << jsonString(a.probeDescendant)
      << ",\"rotation_deg\":" << a.probeAngle << ",\"chain\":[";
    for (size_t i = 0; i < c.chain.size(); ++i) { if (i) o << ','; o << jsonString(c.chain[i]); }
    o << "],\"world_before\":"; jsonMat(o, c.before); o << ",\"world_after\":"; jsonMat(o, c.after);
    o << ",\"max_abs_delta\":" << c.maxDelta << "},\n  \"statistics\":{\"nodes\":" << a.nodes.size()
      << ",\"materials\":" << a.materials.size() << ",\"meshes\":" << a.primitives.size()
      << ",\"triangles\":" << c.triangles << "}\n}\n";
    return o.str();
}
void writeText(const std::string& path, const std::string& value) {
    std::ofstream out(path.c_str(), std::ios::binary);
    require(bool(out), "Cannot write report: " + path);
    out.write(value.data(), static_cast<std::streamsize>(value.size()));
    require(bool(out), "Cannot finish report: " + path);
}

GLuint shader(GLenum type, const char* source) {
    GLuint s = glCreateShader(type); glShaderSource(s, 1, &source, NULL); glCompileShader(s);
    GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[2048] = {}; glGetShaderInfoLog(s, sizeof(log), NULL, log); glDeleteShader(s); throw std::runtime_error(std::string("Shader compile failed: ") + log); }
    return s;
}
GLuint program() {
    const char* vs = R"GLSL(#version 330 core
layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;
layout(location=2) in vec2 texcoord;
uniform mat4 model;
uniform mat4 viewProjection;
uniform mat3 normalMatrix;
out vec3 worldNormal;
out vec2 uv;
void main() {
    gl_Position = viewProjection * model * vec4(position, 1.0);
    worldNormal = normalMatrix * normal;
    uv = texcoord;
})GLSL";
    const char* fs = R"GLSL(#version 330 core
in vec3 worldNormal;
in vec2 uv;
uniform sampler2D baseTexture;
uniform bool textured;
uniform bool maskMode;
uniform float alphaCutoff;
uniform vec4 baseColor;
out vec4 color;
vec3 toSrgb(vec3 x) {
    x = max(x, vec3(0.0));
    return mix(12.92*x, 1.055*pow(x,vec3(1.0/2.4))-0.055, step(vec3(0.0031308),x));
}
void main() {
    vec4 base = baseColor * (textured ? texture(baseTexture,uv) : vec4(1.0));
    if (maskMode && base.a < alphaCutoff) discard;
    vec3 n = normalize(worldNormal) * (gl_FrontFacing ? 1.0 : -1.0);
    float diffuse = max(dot(n,normalize(vec3(0.5,0.85,0.35))),0.0);
    vec3 lit = base.rgb * (0.48 + 0.52*diffuse);
    color = vec4(toSrgb(lit),1.0);
})GLSL";
    GLuint v = shader(GL_VERTEX_SHADER, vs), f = shader(GL_FRAGMENT_SHADER, fs);
    GLuint p = glCreateProgram(); glAttachShader(p,v); glAttachShader(p,f); glLinkProgram(p);
    glDeleteShader(v); glDeleteShader(f);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[2048] = {}; glGetProgramInfoLog(p,sizeof(log),NULL,log); glDeleteProgram(p); throw std::runtime_error(std::string("Shader link failed: ") + log); }
    return p;
}
struct GpuMesh { GLuint vao, vbo; GLsizei count; };
struct Input { const Asset* asset; float angle; std::string view; };
void keyCallback(GLFWwindow* w, int key, int, int action, int) {
    if (action != GLFW_PRESS) return;
    Input& input = *static_cast<Input*>(glfwGetWindowUserPointer(w));
    if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(w, GLFW_TRUE);
    // 字母键同时适用于远程键盘和不发送扩展方向键扫描码的输入设备。
    else if (key == GLFW_KEY_LEFT || key == GLFW_KEY_A) input.angle = input.asset->minAngle;
    else if (key == GLFW_KEY_RIGHT || key == GLFW_KEY_D) input.angle = input.asset->maxAngle;
    else if (key == GLFW_KEY_1) input.view = "hero";
    else if (key == GLFW_KEY_2) input.view = "front";
    else if (key == GLFW_KEY_3) input.view = "top";
}

// OpenGL framebuffer 按左下原点读取，PPM 需要逐行反转。
void capturePpm(const std::string& path, int width, int height) {
    require(width > 0 && height > 0, "Empty framebuffer");
    std::vector<unsigned char> pixels(static_cast<size_t>(width)*height*3);
    glPixelStorei(GL_PACK_ALIGNMENT,1); glReadBuffer(GL_BACK); glFinish();
    glReadPixels(0,0,width,height,GL_RGB,GL_UNSIGNED_BYTE,&pixels[0]);
    require(glGetError() == GL_NO_ERROR, "OpenGL framebuffer read failed");
    std::ofstream out(path.c_str(), std::ios::binary);
    require(bool(out), "Cannot write capture: " + path);
    out << "P6\n" << width << ' ' << height << "\n255\n";
    for (int y = height-1; y >= 0; --y)
        out.write(reinterpret_cast<const char*>(&pixels[static_cast<size_t>(y)*width*3]), static_cast<std::streamsize>(width*3));
    require(bool(out), "Cannot finish capture: " + path);
}

void render(const Asset& a, float angle, const std::string& view, const std::string& capture) {
    require(glfwInit() == GLFW_TRUE, "GLFW initialization failed");
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3); glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES,4);
    if (!capture.empty()) glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow* w = glfwCreateWindow(1280,800,u8"2024270223_王锦政_期末大作业",NULL,NULL);
    if (!w) { glfwTerminate(); throw std::runtime_error("Cannot create GLFW window"); }
    glfwMakeContextCurrent(w);
    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
        glfwDestroyWindow(w); glfwTerminate(); throw std::runtime_error("Cannot initialize OpenGL functions");
    }
    try {
        GLuint p = program();
        std::vector<GpuMesh> meshes(a.primitives.size());
        for (size_t i = 0; i < meshes.size(); ++i) {
            GpuMesh& g = meshes[i]; const std::vector<Vertex>& v = a.primitives[i].vertices;
            g.count = static_cast<GLsizei>(v.size());
            glGenVertexArrays(1,&g.vao); glGenBuffers(1,&g.vbo);
            glBindVertexArray(g.vao); glBindBuffer(GL_ARRAY_BUFFER,g.vbo);
            glBufferData(GL_ARRAY_BUFFER,static_cast<GLsizeiptr>(v.size()*sizeof(Vertex)),&v[0],GL_STATIC_DRAW);
            glEnableVertexAttribArray(0); glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(Vertex),reinterpret_cast<void*>(offsetof(Vertex,p)));
            glEnableVertexAttribArray(1); glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,sizeof(Vertex),reinterpret_cast<void*>(offsetof(Vertex,n)));
            glEnableVertexAttribArray(2); glVertexAttribPointer(2,2,GL_FLOAT,GL_FALSE,sizeof(Vertex),reinterpret_cast<void*>(offsetof(Vertex,uv)));
        }
        std::vector<GLuint> textures(a.materials.size(),0);
        for (size_t i = 0; i < textures.size(); ++i) if (a.materials[i].texture != "-") {
            const Texture& t = a.materials[i].image;
            glGenTextures(1,&textures[i]); glBindTexture(GL_TEXTURE_2D,textures[i]);
            glPixelStorei(GL_UNPACK_ALIGNMENT,1);
            // RGB 按 sRGB 解码，alpha 保持线性；片元输出再显式编码 sRGB。
            glTexImage2D(GL_TEXTURE_2D,0,GL_SRGB8_ALPHA8,t.width,t.height,0,GL_RGBA,GL_UNSIGNED_BYTE,&t.rgba[0]);
            glGenerateMipmap(GL_TEXTURE_2D);
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_LINEAR);
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_REPEAT);
        }
        require(glGetError() == GL_NO_ERROR, "OpenGL resource upload failed");
        Input input = { &a, angle, view }; glfwSetWindowUserPointer(w,&input); glfwSetKeyCallback(w,keyCallback);
        glEnable(GL_DEPTH_TEST); glDisable(GL_BLEND); glDisable(GL_FRAMEBUFFER_SRGB);
        glUseProgram(p); glUniform1i(glGetUniformLocation(p,"baseTexture"),0);
        int frame = 0;
        while (!glfwWindowShouldClose(w)) {
            int width, height; glfwGetFramebufferSize(w,&width,&height);
            if (width <= 0 || height <= 0) { glfwPollEvents(); continue; }
            glViewport(0,0,width,height); glClearColor(0.93f,0.92f,0.89f,1); glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
            const View& v = a.views.find(input.view)->second;
            glm::vec3 direction = glm::normalize(v.target-v.eye), up(0,1,0);
            if (glm::length(glm::cross(direction,up)) < 1e-5f) up = glm::vec3(0,0,-1);
            // 按观察距离设置近平面，避免建筑远视时毫米级接缝发生深度量化冲突。
            const float distance=glm::length(v.target-v.eye);
            const float nearPlane=std::max(0.05f,distance*0.007f);
            glm::mat4 vp = glm::perspective(glm::radians(45.0f),float(width)/height,nearPlane,std::max(100.0f,distance*20.0f)) * glm::lookAt(v.eye,v.target,up);
            glUniformMatrix4fv(glGetUniformLocation(p,"viewProjection"),1,GL_FALSE,glm::value_ptr(vp));
            const std::vector<glm::mat4> world = worldMatrices(a,a.interactionIndex,a.axis,input.angle);
            for (size_t i = 0; i < meshes.size(); ++i) {
                const Primitive& item = a.primitives[i]; const Material& m = a.materials[item.materialIndex];
                const glm::mat4& model = world[item.nodeIndex];
                const glm::mat3 normal = glm::transpose(glm::inverse(glm::mat3(model)));
                glUniformMatrix4fv(glGetUniformLocation(p,"model"),1,GL_FALSE,glm::value_ptr(model));
                glUniformMatrix3fv(glGetUniformLocation(p,"normalMatrix"),1,GL_FALSE,glm::value_ptr(normal));
                glUniform4fv(glGetUniformLocation(p,"baseColor"),1,glm::value_ptr(m.color));
                glUniform1i(glGetUniformLocation(p,"textured"),textures[item.materialIndex] != 0);
                glUniform1i(glGetUniformLocation(p,"maskMode"),m.mask);
                glUniform1f(glGetUniformLocation(p,"alphaCutoff"),m.cutoff);
                glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D,textures[item.materialIndex]);
                glFrontFace(glm::determinant(glm::mat3(model)) < 0 ? GL_CW : GL_CCW);
                if (m.doubleSided) glDisable(GL_CULL_FACE); else { glEnable(GL_CULL_FACE); glCullFace(GL_BACK); }
                glBindVertexArray(meshes[i].vao); glDrawArrays(GL_TRIANGLES,0,meshes[i].count);
            }
            ++frame;
            if (!capture.empty() && frame >= 3) { capturePpm(capture,width,height); break; }
            glfwSwapBuffers(w); glfwPollEvents();
        }
        for (size_t i = 0; i < meshes.size(); ++i) { glDeleteBuffers(1,&meshes[i].vbo); glDeleteVertexArrays(1,&meshes[i].vao); }
        for (size_t i = 0; i < textures.size(); ++i) if (textures[i]) glDeleteTextures(1,&textures[i]);
        glDeleteProgram(p);
    } catch (...) { glfwDestroyWindow(w); glfwTerminate(); throw; }
    glfwDestroyWindow(w); glfwTerminate();
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string root, capture, reportPath, view = "hero";
        bool selfTest = false, hasAngle = false; float angle = 0;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--self-test") selfTest = true;
            else if (arg == "--asset" || arg == "--capture" || arg == "--report" || arg == "--angle" || arg == "--view") {
                require(i+1 < argc, "Missing value for " + arg); const std::string value = argv[++i];
                require(!value.empty(), "Empty value for " + arg);
                if (arg == "--asset") root = value;
                else if (arg == "--capture") capture = value;
                else if (arg == "--report") reportPath = value;
                else if (arg == "--angle") { angle = number(value,"--angle"); hasAngle = true; }
                else view = value;
            } else throw std::runtime_error("Unknown argument: " + arg);
        }
        require(!root.empty(), "Usage: campus-asset-probe --asset <package_dir> [--self-test] [--angle deg] [--view hero|front|top] [--report file.json] [--capture file.ppm]");
        require(!(selfTest && !capture.empty()), "--self-test cannot be combined with --capture");
        const Asset a = loadAsset(root);
        require(a.views.count(view) != 0, "View is not declared by the asset: " + view);
        if (!hasAngle) angle = a.minAngle;
        const Check result = checkAsset(a,angle);
        const std::string output = report(a,result,angle);
        if (!reportPath.empty()) writeText(reportPath,output);
        if (selfTest) std::cout << output;
        else {
            std::cout << "Asset verified. A/D or left/right: minimum/maximum angle; 1/2/3: hero/front/top; Esc: exit.\n";
            render(a,angle,view,capture);
            if (!capture.empty()) std::cout << "Capture saved: " << capture << '\n';
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Asset probe failed: " << e.what() << '\n';
        return 1;
    }
}
