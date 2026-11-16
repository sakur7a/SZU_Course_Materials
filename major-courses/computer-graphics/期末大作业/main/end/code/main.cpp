#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Shader.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#endif

namespace {
const float PI = 3.14159265358979323846f;
const int SHADOW_SIZE = 2048;
const int SOUTH_SPINE_VIEW = 119;
const int NEW_ZHAI_ARCADE_VIEW = 120;
const int ZHOUNAN_ZHAI_VIEW = 121;
const int ZHOUNAN_ZHAI_INSIDE_VIEW = 122;
const int ZHAI_ROADSIDE_STAIR_VIEW = 123;

// 参考资料用于近似造型；坐标是教学建模坐标，不是实地测绘数据。
struct Vertex { glm::vec3 p, n; glm::vec2 uv; };
static_assert(sizeof(Vertex) == 8 * sizeof(float), "Unexpected packed mesh vertex layout");

void vertex(std::vector<Vertex>& out, const glm::vec3& p, const glm::vec3& n, float u, float v) {
    Vertex value = {p, n, glm::vec2(u, v)};
    out.push_back(value);
}
void quad(std::vector<Vertex>& out, const glm::vec3& a, const glm::vec3& b,
          const glm::vec3& c, const glm::vec3& d, const glm::vec3& n) {
    vertex(out, a, n, 0, 0); vertex(out, b, n, 1, 0); vertex(out, c, n, 1, 1);
    vertex(out, a, n, 0, 0); vertex(out, c, n, 1, 1); vertex(out, d, n, 0, 1);
}

std::vector<Vertex> cubeGeometry() {
    std::vector<Vertex> v;
    const float s = 0.5f;
    quad(v, {-s,-s,s},{s,-s,s},{s,s,s},{-s,s,s},{0,0,1});
    quad(v, {s,-s,-s},{-s,-s,-s},{-s,s,-s},{s,s,-s},{0,0,-1});
    quad(v, {s,-s,s},{s,-s,-s},{s,s,-s},{s,s,s},{1,0,0});
    quad(v, {-s,-s,-s},{-s,-s,s},{-s,s,s},{-s,s,-s},{-1,0,0});
    quad(v, {-s,s,s},{s,s,s},{s,s,-s},{-s,s,-s},{0,1,0});
    quad(v, {-s,-s,-s},{s,-s,-s},{s,-s,s},{-s,-s,s},{0,-1,0});
    return v;
}

// 汇星楼屋顶玻璃塔的金属框架合并成一份网格，避免每根框条单独发起绘制。
std::vector<Vertex> huixingFrameGeometry() {
    const float centerX=-.51f, centerZ=1.53f, width=12.44f, depth=24.17f;
    const float bottom=72.5f, top=95.0f;
    std::vector<Vertex> out;
    const std::vector<Vertex> unit=cubeGeometry();
    auto addBox = [&](glm::vec3 center, glm::vec3 size) {
        for (size_t i=0;i<unit.size();++i) {
            Vertex p=unit[i];
            p.p=center+p.p*size;
            out.push_back(p);
        }
    };
    for (int i=0;i<=8;++i) {
        float x=centerX-width*.5f+width*i/8.0f;
        for (int sign=-1;sign<=1;sign+=2)
            addBox({x,(bottom+top)*.5f,centerZ+sign*depth*.5f},{.11f,top-bottom,.11f});
    }
    for (int i=0;i<=16;++i) {
        float z=centerZ-depth*.5f+depth*i/16.0f;
        for (int sign=-1;sign<=1;sign+=2)
            addBox({centerX+sign*width*.5f,(bottom+top)*.5f,z},{.11f,top-bottom,.11f});
    }
    for (int i=0;i<=10;++i) {
        float y=bottom+(top-bottom)*i/10.0f;
        for (int sign=-1;sign<=1;sign+=2) {
            addBox({centerX,y,centerZ+sign*depth*.5f},{width,.11f,.11f});
            addBox({centerX+sign*width*.5f,y,centerZ},{.11f,.11f,depth});
        }
    }
    return out;
}

// 北馆四面窄窗的横向分格合并为一个网格，保持细节而不逐根发起绘制。
std::vector<Vertex> northLibraryWindowGeometry() {
    std::vector<Vertex> out;
    const std::vector<Vertex> unit=cubeGeometry();
    auto addBox = [&](glm::vec3 center, glm::vec3 size) {
        for (size_t i=0;i<unit.size();++i) {
            Vertex p=unit[i]; p.p=center+p.p*size; out.push_back(p);
        }
    };
    for (int floor=0;floor<5;++floor) {
        const float y=6.4f+floor*3.55f;
        for (int i=0;i<15;++i) {
            const float p=-26.32f+i*(56.4f/15.0f);
            for (int side=-1;side<=1;side+=2) {
                // 二层入口玻璃后退，原前排横窗框不能悬在新灰空间中。
                if (floor>0 || side<0) addBox({p,y,side*29.91f},{1.42f,.27f,.13f});
                // 侧面二层的通道门洞不被窗框封住。
                if (floor>0 || ((p<13.0f || p>20.8f) && p<22.5f))
                    addBox({side*29.91f,y,p},{.13f,.27f,1.42f});
            }
        }
    }
    return out;
}

std::vector<Vertex> cylinderGeometry() {
    std::vector<Vertex> v;
    const int segments = 36;
    for (int i = 0; i < segments; ++i) {
        float a = 2 * PI * i / segments, b = 2 * PI * (i + 1) / segments;
        glm::vec3 pa(0.5f * cos(a), -0.5f, 0.5f * sin(a));
        glm::vec3 pb(0.5f * cos(b), -0.5f, 0.5f * sin(b));
        glm::vec3 na(cos(a), 0, sin(a)), nb(cos(b), 0, sin(b));
        float u0 = float(i) / segments, u1 = float(i + 1) / segments;
        vertex(v, pa, na, u0, 0); vertex(v, pb, nb, u1, 0); vertex(v, pb + glm::vec3(0,1,0), nb, u1, 1);
        vertex(v, pa, na, u0, 0); vertex(v, pb + glm::vec3(0,1,0), nb, u1, 1); vertex(v, pa + glm::vec3(0,1,0), na, u0, 1);
        vertex(v, {0,0.5f,0}, {0,1,0}, .5f, .5f);
        vertex(v, pb + glm::vec3(0,1,0), {0,1,0}, pb.x + .5f, pb.z + .5f);
        vertex(v, pa + glm::vec3(0,1,0), {0,1,0}, pa.x + .5f, pa.z + .5f);
        vertex(v, {0,-.5f,0}, {0,-1,0}, .5f, .5f);
        vertex(v, pa, {0,-1,0}, pa.x + .5f, pa.z + .5f);
        vertex(v, pb, {0,-1,0}, pb.x + .5f, pb.z + .5f);
    }
    return v;
}

// 致远门草坪校名装置的镂空红心：前后面和内外侧均为真实三维几何。
std::vector<Vertex> heartGeometry() {
    std::vector<Vertex> out;
    const int segments=64;
    auto point=[](float t) {
        float s=sin(t);
        return glm::vec2(s*s*s,(13*cos(t)-5*cos(2*t)-2*cos(3*t)-cos(4*t))/16.0f+.12f);
    };
    auto p3=[](glm::vec2 p,float z) { return glm::vec3(p.x,p.y,z); };
    for (int i=0;i<segments;++i) {
        glm::vec2 a=point(2*PI*i/segments),b=point(2*PI*(i+1)/segments);
        glm::vec2 ai=a*.61f,bi=b*.61f,d=b-a;
        glm::vec3 side=glm::normalize(glm::vec3(-d.y,d.x,0));
        quad(out,p3(a,.10f),p3(b,.10f),p3(bi,.10f),p3(ai,.10f),{0,0,1});
        quad(out,p3(b,-.10f),p3(a,-.10f),p3(ai,-.10f),p3(bi,-.10f),{0,0,-1});
        quad(out,p3(a,-.10f),p3(b,-.10f),p3(b,.10f),p3(a,.10f),side);
        quad(out,p3(bi,-.10f),p3(ai,-.10f),p3(ai,.10f),p3(bi,.10f),-side);
    }
    return out;
}

std::vector<Vertex> signPlaneGeometry() {
    std::vector<Vertex> out;
    // 单面字牌避免透明字形从薄盒子的背面重复透出。
    quad(out,{-.5f,-.5f,0},{.5f,-.5f,0},{.5f,.5f,0},{-.5f,.5f,0},{0,0,1});
    return out;
}

std::vector<Vertex> sphereGeometry() {
    std::vector<Vertex> v;
    const int slices = 24, stacks = 14;
    for (int j = 0; j < stacks; ++j) {
        float v0 = float(j) / stacks, v1 = float(j + 1) / stacks;
        for (int i = 0; i < slices; ++i) {
            float u0 = float(i) / slices, u1 = float(i + 1) / slices;
            glm::vec3 p[4];
            float u[4] = {u0,u1,u1,u0}, t[4] = {v0,v0,v1,v1};
            for (int k = 0; k < 4; ++k) {
                float az = 2 * PI * u[k], el = PI * (t[k] - .5f);
                p[k] = glm::vec3(cos(el) * cos(az), sin(el), cos(el) * sin(az));
            }
            int ids[] = {0,1,2,0,2,3};
            for (int k = 0; k < 6; ++k) { int n = ids[k]; vertex(v, p[n] * .5f, p[n], u[n], t[n]); }
        }
    }
    return v;
}

// 以不规则树冠团承载叶片照片材质；树种与逐棵位置仍是近似。
std::vector<Vertex> distantTreeCrown(int slices, int stacks, float crownScale=1.0f) {
    std::vector<Vertex> v;
    const glm::vec3 centers[]={{0,4.5f,0},{-1.1f,4.2f,.1f},{1.1f,4.35f,.15f},{0,4.9f,-.85f}};
    const glm::vec3 sizes[]={{4.3f,2.8f,4.1f},{3.0f,2.4f,3.1f},{3.1f,2.5f,3.0f},{3.2f,2.2f,3.0f}};
    for (int c=0;c<4;++c) for (int j=0;j<stacks;++j) for (int i=0;i<slices;++i) {
        const int ids[]={0,1,2,0,2,3};
        const float u[]={float(i)/slices,float(i+1)/slices,float(i+1)/slices,float(i)/slices};
        const float t[]={float(j)/stacks,float(j)/stacks,float(j+1)/stacks,float(j+1)/stacks};
        glm::vec3 p[4],n[4];
        for (int k=0;k<4;++k) {
            float az=2*PI*u[k], el=PI*(t[k]-.5f);
            n[k]=glm::vec3(cos(el)*cos(az),sin(el),cos(el)*sin(az));
            float irregular=1.0f+.08f*sin(3*az+2*el+c)+.045f*cos(5*az-3*el);
            p[k]=centers[c]+n[k]*sizes[c]*(.5f*irregular*crownScale);
        }
        for (int k=0;k<6;++k) { int q=ids[k]; vertex(v,p[q],n[q],u[q],t[q]); }
    }
    return v;
}

// 透明枝叶簇从不同方向伸出冠层，打破球体轮廓；远处使用较少的面片。
std::vector<Vertex> treeLeafCards(int layers) {
    std::vector<Vertex> v;
    if (layers == 3) {
        // 近处用多层互相遮挡的枝叶簇遮住小尺寸内冠，避免光滑球体和露出树顶。
        const float heights[]={3.25f,3.80f,4.35f,4.90f,5.45f,6.00f};
        const float radii[]  ={1.15f,1.80f,2.12f,2.05f,1.48f,.62f};
        const int counts[]   ={7,10,12,12,10,6};
        for (int layer=0;layer<6;++layer) for (int i=0;i<counts[layer];++i) {
            float a=2.0f*PI*(i+layer*.37f)/counts[layer];
            float jitter=.13f*sin(i*5.13f+layer*2.17f);
            glm::vec3 outward(cos(a),0,sin(a));
            glm::vec3 center=outward*(radii[layer]+jitter)+glm::vec3(0,heights[layer]+jitter,0);
            glm::vec3 right(-sin(a),0,cos(a));
            float tilt=(layer-2.5f)*.30f;
            glm::vec3 up=glm::normalize(glm::vec3(outward.x*tilt,1,outward.z*tilt));
            glm::vec3 normal=glm::normalize(glm::cross(right,up));
            float halfWidth=1.22f+.10f*sin(i*2.7f+layer);
            float halfHeight=.92f+.09f*cos(i*1.9f+layer);
            quad(v,center-right*halfWidth-up*halfHeight,center+right*halfWidth-up*halfHeight,
                 center+right*halfWidth+up*halfHeight,center-right*halfWidth+up*halfHeight,normal);
        }
        // 几张近水平的枝叶簇覆盖顶部，避免俯视时看到光滑内冠和树枝尖。
        for (int i=0;i<7;++i) {
            float a=2.0f*PI*i/7.0f;
            glm::vec3 radial(cos(a),0,sin(a)),across(-sin(a),0,cos(a));
            glm::vec3 along=glm::normalize(glm::vec3(radial.x,-.45f,radial.z));
            glm::vec3 center=radial*.65f+glm::vec3(0,6.32f+.04f*sin(i*4.0f),0);
            glm::vec3 normal=glm::normalize(glm::cross(across,along));
            quad(v,center-along*1.05f-across*1.05f,center+along*1.05f-across*1.05f,
                 center+along*1.05f+across*1.05f,center-along*1.05f+across*1.05f,normal);
        }
        return v;
    }
    const int count=7;
    for (int layer=0;layer<layers;++layer) for (int i=0;i<count;++i) {
        float a=2.0f*PI*(i+layer*.42f)/count;
        float radius=layer==0 ? 1.65f : layer==1 ? 2.0f : 1.1f;
        float height=layer==0 ? 3.5f : layer==1 ? 4.6f : 5.5f;
        glm::vec3 center(radius*cos(a),height,radius*sin(a));
        glm::vec3 right(-sin(a),0,cos(a));
        glm::vec3 up=glm::normalize(glm::vec3(.32f*cos(a),.90f,.32f*sin(a)));
        glm::vec3 normal=glm::normalize(glm::cross(right,up));
        float width=layer==2 ? 2.7f : 3.1f, halfHeight=layer==2 ? 1.0f : 1.1f;
        quad(v,center-right*width*.5f-up*halfHeight,center+right*width*.5f-up*halfHeight,
             center+right*width*.5f+up*halfHeight,center-right*width*.5f+up*halfHeight,normal);
    }
    return v;
}

// 细长棕榈叶向四周下垂；轮廓参考南馆前实景，非逐株测绘。
std::vector<Vertex> royalPalmCrown() {
    std::vector<Vertex> v;
    for (int frond=0;frond<9;++frond) {
        float a=frond*2*PI/9.0f;
        glm::vec3 outward(cos(a),0,sin(a)),side(-sin(a),0,cos(a));
        for (int pair=0;pair<7;++pair) {
            float t=(pair+.5f)/7.0f;
            glm::vec3 center=outward*(2.65f*t)+glm::vec3(0,.25f*(1-t)-.95f*t*t,0);
            for (int sign=-1;sign<=1;sign+=2) {
                glm::vec3 p=center-outward*.13f;
                glm::vec3 q=center+outward*.28f;
                glm::vec3 tip=center+outward*.25f+side*(sign*.63f*(1-.48f*t))
                                   +glm::vec3(0,-.30f-.16f*t,0);
                glm::vec3 normal=glm::normalize(glm::cross(q-p,tip-p));
                vertex(v,p,normal,0,0); vertex(v,q,normal,1,0); vertex(v,tip,normal,.5f,1);
            }
        }
    }
    return v;
}

std::vector<Vertex> distantTreeTrunk() {
    std::vector<Vertex> v=cylinderGeometry();
    for (size_t i=0;i<v.size();++i) {
        v[i].p.x*=.55f; v[i].p.y=v[i].p.y*4.0f+2.0f; v[i].p.z*=.55f;
    }
    return v;
}

// 标准圆环：主半径 0.5，供浑天仪、车轮和广场边线复用。
std::vector<Vertex> torusGeometry() {
    std::vector<Vertex> v;
    const int major = 56, minor = 10;
    for (int i = 0; i < major; ++i) for (int j = 0; j < minor; ++j) {
        int ids[] = {0,1,2,0,2,3};
        int di[] = {0,1,1,0}, dj[] = {0,0,1,1};
        glm::vec3 p[4], n[4];
        float u[4], w[4];
        for (int k = 0; k < 4; ++k) {
            u[k] = float(i + di[k]) / major; w[k] = float(j + dj[k]) / minor;
            float a = 2 * PI * u[k], b = 2 * PI * w[k];
            n[k] = glm::vec3(cos(a)*cos(b), sin(b), sin(a)*cos(b));
            p[k] = glm::vec3(.5f*cos(a),0,.5f*sin(a)) + n[k] * .028f;
        }
        for (int k = 0; k < 6; ++k) { int q = ids[k]; vertex(v, p[q], n[q], u[q], w[q]); }
    }
    return v;
}

// 读取离线转换后的三角形网格，顶点包含位置、法线及纹理坐标。
std::vector<Vertex> readMesh(const std::string& path, bool allowEmpty = false) {
    std::ifstream file(path.c_str(), std::ios::binary);
    char magic[4] = {0,0,0,0};
    std::uint32_t count = 0;
    file.read(magic, 4);
    file.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!file || magic[0]!='S' || magic[1]!='Z' || magic[2]!='M' || magic[3]!='1' || (!allowEmpty && count == 0) || count > 500000)
        throw std::runtime_error("Cannot read mesh: " + path);
    std::vector<Vertex> vertices(count);
    if (count) file.read(reinterpret_cast<char*>(&vertices[0]), static_cast<std::streamsize>(count * sizeof(Vertex)));
    if (!file) throw std::runtime_error("Truncated mesh: " + path);
    return vertices;
}

class Mesh {
public:
    GLuint vao, vbo;
    GLsizei count;
    explicit Mesh(const std::vector<Vertex>& vertices) : vao(0), vbo(0), count(static_cast<GLsizei>(vertices.size())) {
        glGenVertexArrays(1, &vao); glGenBuffers(1, &vbo);
        glBindVertexArray(vao); glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.empty() ? NULL : &vertices[0], GL_STATIC_DRAW);
        glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex,p)));
        glEnableVertexAttribArray(1); glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex,n)));
        glEnableVertexAttribArray(2); glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex,uv)));
        glBindVertexArray(0);
    }
    ~Mesh() { glDeleteBuffers(1, &vbo); glDeleteVertexArrays(1, &vao); }
    void draw() const { glBindVertexArray(vao); glDrawArrays(GL_TRIANGLES, 0, count); }
private:
    Mesh(const Mesh&);
    Mesh& operator=(const Mesh&);
};

struct Image { int width, height; std::vector<unsigned char> pixels; };
Image readPPM(const std::string& path) {
    std::ifstream file(path.c_str(), std::ios::binary);
    std::string magic; int w = 0, h = 0, maxValue = 0;
    file >> magic >> w >> h >> maxValue;
    if (!file || magic != "P6" || w <= 0 || h <= 0 || w > 4096 || h > 4096 || maxValue != 255)
        throw std::runtime_error("Cannot read PPM texture: " + path);
    file.get();
    Image result; result.width = w; result.height = h;
    result.pixels.resize(static_cast<size_t>(w) * h * 3);
    file.read(reinterpret_cast<char*>(&result.pixels[0]), static_cast<std::streamsize>(result.pixels.size()));
    if (!file) throw std::runtime_error("Truncated PPM texture: " + path);
    return result;
}

GLuint loadTexture(const std::string& path) {
    Image img = readPPM(path);
    GLuint texture = 0; glGenTextures(1, &texture); glBindTexture(GL_TEXTURE_2D, texture);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, img.width, img.height, 0, GL_RGB, GL_UNSIGNED_BYTE, &img.pixels[0]);
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    return texture;
}

// 树叶纹理保留 alpha，着色器用透明裁剪显示叶片轮廓。
GLuint loadRgbaTexture(const std::string& path, bool palette = false) {
    std::ifstream file(path.c_str(), std::ios::binary);
    std::uint32_t w = 0, h = 0;
    file.read(reinterpret_cast<char*>(&w), sizeof(w)); file.read(reinterpret_cast<char*>(&h), sizeof(h));
    if (!file || !w || !h || w > 4096 || h > 4096) throw std::runtime_error("Cannot read RGBA texture: " + path);
    std::vector<unsigned char> pixels(static_cast<size_t>(w) * h * 4);
    file.read(reinterpret_cast<char*>(&pixels[0]), static_cast<std::streamsize>(pixels.size()));
    if (!file) throw std::runtime_error("Truncated RGBA texture: " + path);
    GLuint texture = 0; glGenTextures(1,&texture); glBindTexture(GL_TEXTURE_2D,texture);
    glPixelStorei(GL_UNPACK_ALIGNMENT,1);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,w,h,0,GL_RGBA,GL_UNSIGNED_BYTE,&pixels[0]);
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,palette ? GL_NEAREST : GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,palette ? GL_NEAREST : GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_REPEAT);
    return texture;
}

#include "CampusModelLayers.h"
#include "CampusAsset.h"
#include "CampusGates.h"
#include "CampusPlaces.h"

// 运行目录设为 exe 所在目录，使双击或从其他当前目录启动都能定位资源。
void enterExecutableDirectory() {
#ifdef _WIN32
    std::vector<wchar_t> path(32768);
    DWORD length = GetModuleFileNameW(NULL, &path[0], static_cast<DWORD>(path.size()));
    if (!length || length >= path.size()) throw std::runtime_error("Cannot locate executable directory");
    std::wstring value(&path[0], length);
    size_t last = value.find_last_of(L"\\/");
    if (last == std::wstring::npos || !SetCurrentDirectoryW(value.substr(0, last).c_str()))
        throw std::runtime_error("Cannot enter executable directory");
    SetConsoleOutputCP(CP_UTF8);
#endif
}

struct State {
    float yaw, pitch, speed, armillaryAngle, busDistance, swanTime, kaifengDoor;
    float lideAngle,lideMaxAngle;
    glm::vec3 position;
    bool dragging, shadows, groundFollow, busFollow, busAuto, swanFollow, swansActive, kaifengDoorOpen,placeNames;
    bool lideAvailable,lideLowered;
    int gateSelection;
    double lastX, lastY;
    State() : yaw(-90), pitch(0), speed(45), armillaryAngle(0), busDistance(20), swanTime(0), kaifengDoor(0),lideAngle(0),lideMaxAngle(0),
              position(0,80,300), dragging(false), shadows(true), groundFollow(false), busFollow(false), busAuto(false), swanFollow(false), swansActive(true), kaifengDoorOpen(false),placeNames(true),lideAvailable(false),lideLowered(false),gateSelection(-1), lastX(0), lastY(0) {}
};

glm::vec3 cameraDirection(const State& s) {
    float y = glm::radians(s.yaw), p = glm::radians(s.pitch);
    return glm::normalize(glm::vec3(cos(y)*cos(p), sin(p), sin(y)*cos(p)));
}

void aimCamera(State& s, glm::vec3 eye, glm::vec3 target) {
    s.position = eye;
    glm::vec3 direction = glm::normalize(target - eye);
    s.yaw = glm::degrees(std::atan2(direction.z, direction.x));
    s.pitch = glm::degrees(std::asin(glm::clamp(direction.y, -1.0f, 1.0f)));
}

const std::vector<GateView>& campusGateViews() {
    static const std::vector<GateView> views=readGateViews();
    return views;
}

void selectGateView(State& s,int index) {
    const std::vector<GateView>& views=campusGateViews();
    if (views.empty()) throw std::runtime_error("Campus gate views are missing");
    s.gateSelection=(index%static_cast<int>(views.size())+static_cast<int>(views.size()))%static_cast<int>(views.size());
    const GateView& gate=views[s.gateSelection];
    s.groundFollow=false;s.busFollow=false;s.swanFollow=false;s.speed=10;
    aimCamera(s,gate.eye,gate.target);std::cout<<gate.name<<"\n";
}

void resetView(State& s, int view) {
    // 坐标单位约为米；四个快捷位置使用的是同一个可自由移动的相机。
    s.groundFollow = false;
    s.busFollow = false;
    s.swanFollow = false;
    if (view == 1) { aimCamera(s,{-650,2100,680},{-20,10,30}); s.speed = 170; }
    if (view == 2) { aimCamera(s,{-90,74,110},{-114,28,-45}); s.speed = 32; }
    if (view == 3) { aimCamera(s,{395,190,-700},{420,12,-405}); s.speed = 60; }
    if (view == 4) { aimCamera(s,{650,235,835},{290,25,430}); s.speed = 70; }
    if (view == 5) { aimCamera(s,{95,135,-165},{-12,64,-300}); s.speed = 45; }
    if (view == 6) { aimCamera(s,{-440,105,380},{-300,10,205}); s.speed = 40; }
    if (view == 7) { aimCamera(s,{100,60,425},{185,10,340}); s.speed = 28; }
    if (view == 8) { aimCamera(s,{525,100,70},{410,13,135}); s.speed = 36; }
    if (view == 9) { aimCamera(s,{85,95,-500},{205,18,-375}); s.speed = 36; }
    if (view == 10) { aimCamera(s,{-100,85,840},{12,28,715}); s.speed = 30; }
    if (view == 11) { aimCamera(s,{-390,115,-395},{-175,25,-545}); s.speed = 36; }
    if (view == 12) { aimCamera(s,{240,105,830},{124,34,751}); s.speed = 30; }
    if (view == 13) { aimCamera(s,{25,60,-80},{-42,22,-156}); s.speed = 25; }
    if (view == 14) { aimCamera(s,{-360,85,-180},{-257,34,-126}); s.speed = 25; }
    if (view == 15) { aimCamera(s,{80,75,-170},{172,14,-247}); s.speed = 25; }
    if (view == 16) { aimCamera(s,{30,60,325},{140,9,247}); s.speed = 25; }
    if (view == 17) { aimCamera(s,{80,70,-625},{166,15,-556}); s.speed = 25; }
    if (view == 18) { aimCamera(s,{325,72,275},{435,26,348}); s.speed = 25; }
    if (view == 19) { aimCamera(s,{-40,85,630},{105,20,550}); s.speed = 25; }
    if (view == 20) { aimCamera(s,{-280,41,-22},{-243,25,-49}); s.speed = 18; }
    if (view == 21) { aimCamera(s,{-510,145,690},{-330,38,577}); s.speed = 35; }
    if (view == 22) { aimCamera(s,{430,115,-80},{310,35,-185}); s.speed = 30; }
    if (view == 23) { aimCamera(s,{-443,20.5f,-54},{-410,23.5f,-54}); s.speed = 12; }
    if (view == 24) { aimCamera(s,{328,35,-375},{291,18,-400}); s.speed = 18; }
    if (view == 25) { aimCamera(s,{-102,36,-14},{-123,33,-84}); s.speed = 18; }
    if (view == 26) { aimCamera(s,{54,72,188},{123,15,105}); s.speed = 22; }
    if (view == 27) { aimCamera(s,{222,52,137},{165,13,75}); s.speed = 20; }
    if (view == 28) { aimCamera(s,{-90,39,-496},{-130,39,-520}); s.speed = 18; }
    if (view == 29) { aimCamera(s,{438,8,125},{413,10,139}); s.speed = 18; }
    if (view == 30) { aimCamera(s,{168,10.2f,324},{205,10.2f,361}); s.speed = 12; }
    if (view == 31) { aimCamera(s,{409.4f,14.4f,138},{382,14.4f,85}); s.speed = 12; }
    if (view == 32) { aimCamera(s,{-110,115,-465},{-12,70,-300}); s.speed = 24; }
    if (view == 33) { aimCamera(s,{-345,14.5f,225},{-377,10.0f,190}); s.speed = 12; }
    if (view == 34) { aimCamera(s,{-262,14.0f,192},{-231,10.0f,211}); s.speed = 12; }
    if (view == 35) { aimCamera(s,{-90,40,58},{-69.5f,32,129.8f}); s.speed = 18; }
    if (view == 36) { aimCamera(s,{187,55,270},{313,22,349}); s.speed = 22; }
    if (view == 37) { aimCamera(s,{405,19,325},{333,21,318}); s.speed = 14; }
    if (view == 38) { aimCamera(s,{288,8.0f,328},{305,23,344}); s.speed = 12; }
    if (view == 39) { aimCamera(s,{325,72,275},{435,26,348}); s.speed = 20; }
    if (view == 40) { aimCamera(s,{350,40,318},{403,16,371}); s.speed = 14; }
    if (view == 41) { aimCamera(s,{380,16,333},{404,11,372}); s.speed = 12; }
    if (view == 42) { aimCamera(s,{275,75,400},{382,24,458}); s.speed = 20; }
    if (view == 43) { aimCamera(s,{380,22,320},{392,27,390}); s.speed = 14; }
    if (view == 44) { aimCamera(s,{393,6,335},{372,9,370}); s.speed = 12; }
    if (view == 45) { aimCamera(s,{423,5,378},{401,8,400}); s.speed = 12; }
    if (view == 46) { aimCamera(s,{220,155,-490},{50,35,-640}); s.speed = 28; }
    if (view == 47) { aimCamera(s,{-105,130,-790},{50,35,-640}); s.speed = 28; }
    if (view == 48) { aimCamera(s,{52,30,-720},{52,36,-660}); s.speed = 14; }
    if (view == 49) { aimCamera(s,{270,72,-520},{195,38,-650}); s.speed = 22; }
    if (view == 50) { aimCamera(s,{-103,90,18},{-126,38,-100}); s.speed = 24; }
    if (view == 51) { aimCamera(s,{-165,43,-50},{-163,35,-78}); s.speed = 14; }
    if (view == 52) { aimCamera(s,{-75,43,-58},{-84,35,-94}); s.speed = 14; }
    if (view == 53) { aimCamera(s,{52,145,95},{-20,31,-40}); s.speed = 25; }
    if (view == 54) { aimCamera(s,{-43,28.7f,25},{-16,29.7f,15}); s.speed = 14; }
    if (view == 55) { aimCamera(s,{-239,33,-42},{-219,31,-24}); s.speed = 14; }
    if (view == 56) { aimCamera(s,{-220.6f,29.4f,-24},{-221.1f,29.5f,-27.2f}); s.speed = 8; }
    if (view == 57) { aimCamera(s,{-244,31,-70},{-256,36,-126}); s.speed = 14; }
    if (view == 58) { aimCamera(s,{-253.5f,34,-110},{-260.6f,33.5f,-145}); s.speed = 12; }
    if (view == 59) { aimCamera(s,{439.7f,11.2f,213.3f},{438.6f,12.4f,204.7f}); s.speed = 10; }
    if (view == 60) { aimCamera(s,{434.7f,17.5f,179.0f},{409,17.8f,138}); s.speed = 12; }
    if (view == 61) { aimCamera(s,{297,23,-401},{243,24,-385}); s.speed = 12; }
    if (view == 62) { aimCamera(s,{140,24,-531},{204,21,-469}); s.speed = 14; }
    if (view == 63) { aimCamera(s,{285,120,-490},{442,22,-626}); s.speed = 25; }
    if (view == 64) { aimCamera(s,{369.3f,23.3f,-682.1f},{445,21,-617}); s.speed = 12; }
    if (view == 65) { aimCamera(s,{405,90,-375},{545,56,-493}); s.speed = 22; }
    if (view == 66) { aimCamera(s,{422,12.3f,-404},{346,18,-404}); s.speed = 14; }
    if (view == 67) { aimCamera(s,{285,24.7f,-700},{297,23.4f,-678}); s.speed = 10; }
    if (view == 68) { aimCamera(s,{245,82,-770},{289,20,-655}); s.speed = 22; }
    if (view == 69) { aimCamera(s,{151,4.0f,353},{137,3.3f,337}); s.speed = 10; }
    if (view == 70) { aimCamera(s,{184,4.0f,386},{204,3.3f,406}); s.speed = 10; }
    if (view == 71) { aimCamera(s,{201,4.1f,411},{210,3.1f,421.5f}); s.speed = 10; }
    if (view == 72) { aimCamera(s,{110,95,500},{237,14,592}); s.speed = 24; }
    if (view == 73) { aimCamera(s,{222,4.0f,590},{241,12,603}); s.speed = 10; }
    if (view == 74) { aimCamera(s,{217,75,473},{317,24,522}); s.speed = 20; }
    if (view == 75) { aimCamera(s,{266,4.0f,539},{277.5f,5,525.5f}); s.speed = 10; }
    if (view == 76) { aimCamera(s,{389,30,555},{344,15,512}); s.speed = 12; }
    if (view == 77) { aimCamera(s,{115,42,608},{207,10,687}); s.speed = 16; }
    if (view == 78) { aimCamera(s,{145,5,640},{190,10,673}); s.speed = 10; }
    if (view == 79) { aimCamera(s,{422,12,215},{435,9,236}); s.speed = 10; }
    if (view == 80) { aimCamera(s,{409,13,311},{410,8.5f,278}); s.speed = 10; }
    if (view == 81) { aimCamera(s,{288,19.8f,-590},{286,23.8f,-680}); s.speed = 12; }
    if (view == 82) { aimCamera(s,{-104.8f,24.8f,374.4f},{-103.9f,27.2f,354.5f}); s.speed = 8; }
    if (view == 83) { aimCamera(s,{-149.7f,16.0f,167.5f},{-160.8f,19.7f,184.1f}); s.speed = 10; }
    if (view == 84) { aimCamera(s,{-293,14.8f,523},{-270,14.0f,538}); s.speed = 8; }
    if (view == 85) { aimCamera(s,{-425,20.1f,519},{-431,17.8f,556}); s.speed = 8; }
    if (view == 86) { aimCamera(s,{-280,14.1f,516},{-275,14.0f,502}); s.speed = 8; }
    if (view == 87) { aimCamera(s,{-425,16.5f,590},{-419,16.3f,592}); s.speed = 8; }
    if (view == 88) { aimCamera(s,{-426,18.8f,606},{-426,16.5f,586}); s.speed = 8; }
    if (view == 89) { aimCamera(s,{-423.5f,20.0f,527},{-433,20.1f,527}); s.speed = 8; }
    if (view == 90) { aimCamera(s,{-423.5f,18.5f,552.5f},{-433,18.7f,558}); s.speed = 8; }
    if (view == 91) { aimCamera(s,{-246.4f,26.7f,-1.5f},{-247.1f,27.0f,-8}); s.speed = 8; }
    if (view == 92) { aimCamera(s,{-136.0f,18.2f,117.0f},{-142.7f,16.5f,109.5f}); s.speed = 8; }
    if (view == 93) { aimCamera(s,{-282.3f,18.4f,354.0f},{-290.0f,17.4f,350.0f}); s.speed = 8; }
    if (view == 94) { aimCamera(s,{-297.5f,18.9f,346.0f},{-290.0f,17.5f,350.0f}); s.speed = 8; }
    if (view == 95) { aimCamera(s,{-269.1f,11.2f,180.2f},{-263.1f,9.6f,180.7f}); s.speed = 8; s.swanFollow = true; }
    if (view == 96) { aimCamera(s,{-590.0f,28.5f,546.0f},{-595.0f,20.0f,501.0f}); s.speed = 14; }
    if (view == 97) { aimCamera(s,{-597.8f,15.5f,517.0f},{-600.5f,15.7f,503.8f}); s.speed = 6; }
    if (view == 98) { aimCamera(s,{230,3.8f,634},{175,3.8f,635}); s.speed = 8; }
    if (view == 99) { aimCamera(s,{71.0f,20.8f,31.0f},{76.0f,15.2f,65.0f}); s.speed = 8; }
    if (view == 100) { aimCamera(s,{161,18.4f,-524},{128,23,-527}); s.speed = 8; }
    if (view == 101) { aimCamera(s,{-426,25.1f,-137},{-423,28,-170}); s.speed = 8; }
    if (view == 102) { aimCamera(s,{-329,23.3f,-39.5f},{-278,27.8f,-38.8f}); s.speed = 8; }
    if (view == 103) { aimCamera(s,{-269,29.6f,-65.5f},{-327,22.0f,-66}); s.speed = 8; }
    if (view == 104) { aimCamera(s,{511,4.0f,245.3f},{532.4f,5.5f,229.8f}); s.speed = 8; }
    if (view == 105) { aimCamera(s,{235,17.4f,-44.8f},{272,12.8f,-42.5f}); s.speed = 8; }
    if (view == 106) { aimCamera(s,{508.4f,15.5f,-484.2f},{501.0f,18.5f,-508}); s.speed = 8; }
    if (view == 107) { aimCamera(s,{-276,25.4f,-608.8f},{-251,30,-608.8f}); s.speed = 8; }
    if (view == 108) { aimCamera(s,{-276,33,-332},{-303,29.5f,-305}); s.speed = 8; }
    if (view == 109) { aimCamera(s,{223.8f,15.1f,-435.3f},{185.5f,12.5f,-423.7f}); s.speed = 8; }
    if (view == 110) { aimCamera(s,{-359,31,-111.7f},{-399,22,-90.4f}); s.speed = 8; }
    if (view == 111) { aimCamera(s,{-407,23,-92},{-370,29,-98}); s.speed = 8; }
    if (view == 112) { aimCamera(s,{430,225,650},{210,12,470}); s.speed = 55; }
    if (view == 113) { aimCamera(s,{-760,210,730},{-440,25,540}); s.speed = 45; }
    if (view == 114) { aimCamera(s,{-490,110,670},{-310,45,565}); s.speed = 22; }
    if (view == 115) { aimCamera(s,{415,85,175},{565,25,230}); s.speed = 22; }
    if (view>=116 && view<=118) {
        // 高架入口视角随模型中的实际桥面高程生成，避免维护另一套硬编码高度。
        static const std::vector<GateView> entrances=readGateViews("assets/campus-south-dorms.tsv");
        if (entrances.size()!=3) throw std::runtime_error("Missing south dormitory skywalk views");
        const GateView& entrance=entrances[view-116];
        aimCamera(s,entrance.eye,entrance.target);s.speed=8;
    }
    if (view == SOUTH_SPINE_VIEW) { aimCamera(s,{317,5.81f,397},{367,5.4f,336}); s.speed = 8; }
    if (view == NEW_ZHAI_ARCADE_VIEW) { aimCamera(s,{345.6f,25.0f,-228.6f},{332,26.2f,-205}); s.speed = 7; }
    if (view == ZHOUNAN_ZHAI_VIEW) { aimCamera(s,{282.5f,15.0f,-48.5f},{294,11.7f,-58.8f}); s.speed = 5; }
    if (view == ZHOUNAN_ZHAI_INSIDE_VIEW) { aimCamera(s,{301.3f,13.03f,-61},{290,12.1f,-58.5f}); s.speed = 5; }
    if (view == ZHAI_ROADSIDE_STAIR_VIEW) { aimCamera(s,{381.5f,22.4f,-176.3f},{358.3f,19.1f,-185.6f}); s.speed = 7; }
}

// 同一自由相机沿朝向平面移动，Q/E 控制高度。
void moveCamera(State& s, float distance, bool w, bool a, bool back, bool d, bool down, bool up) {
    float angle=glm::radians(s.yaw);
    glm::vec3 forward(cos(angle),0,sin(angle)), right(-sin(angle),0,cos(angle));
    s.position += forward*distance*float(int(w)-int(back));
    s.position += right*distance*float(int(d)-int(a));
    s.position.y += distance*float(int(up)-int(down));
}

void updateObjects(State& s, float dt, bool left, bool right, bool busLeft, bool busRight, float routeLength) {
    if (s.swansActive) s.swanTime += dt;
    float target=s.kaifengDoorOpen ? 1.0f : 0.0f;
    s.kaifengDoor += glm::clamp(target-s.kaifengDoor,-1.25f*dt,1.25f*dt);
    s.armillaryAngle += 55.0f * dt * (int(right) - int(left));
    int direction=int(busRight)-int(busLeft);
    if (!direction && s.busAuto) direction=1;
    s.busDistance = glm::clamp(s.busDistance + 8.0f * dt * direction, 0.0f, routeLength);
    if (s.busAuto && s.busDistance >= routeLength) s.busAuto=false;
}

void printHelp() {
    std::cout << u8"深圳大学粤海校区（地图数据与实景参考，部分建筑和地形为近似）\n"
              << u8"地图数据 © OpenStreetMap contributors（ODbL），详见 assets/CREDITS.txt\n"
              << u8"按住鼠标左键拖动：转头　W/A/S/D：前后左右　Q/E：下降/上升\n"
              << u8"滚轮：调整飞行速度　Shift：加速　Shift+T：切换贴地步行\n"
              << u8"Ctrl+N：显示/隐藏地点名称　Ctrl+G：下一个校门　Ctrl+Shift+G：上一个校门\n"
              << u8"Ctrl+1：南区总览　Ctrl+2：西南生活区总览\n"
              << u8"Ctrl+3：乔苑宿舍楼群　Ctrl+4：南区宿舍楼群\n"
              << u8"Ctrl+5/6/7：夏筝、秋瑟、冬筑的高架步道入口\n"
              << u8"方向键左/右：旋转广场环形装置　J/L：倒退/前进校园小巴\n"
              << u8"Shift+J：跟随小巴　Shift+L：小巴自动前进/暂停　移动或拖动鼠标退出跟随\n"
              << u8"Shift+F6：跟随黑天鹅　Ctrl+F6：暂停/继续天鹅游动\n"
              << u8"Shift+F4：凯风餐厅全景　Shift+F3：餐厅入口　Ctrl+F4：打开/关闭入口门\n"
              << u8"F1：全校园　F2：天人广场　F3：体育场　F4：南区　F5：汇星楼　F6：文山湖\n"
              << u8"F7：召南桥　F8：周南桥及南区连廊　F9：元平体育馆\n"
              << u8"F10：致腾楼　F11：汇文楼　F12：致原楼\n"
              << u8"数字 1：档案馆　数字 2：演会中心　数字 3：校友广场\n"
              << u8"数字 4：运动广场　数字 5：至柔馆　数字 6：致真楼　数字 7：致信楼\n"
              << u8"数字 8：时光广场日晷　数字 9：乔苑宿舍　数字 0：斋区　Ctrl+0：新斋首层廊道\n"
              << u8"Ctrl+9：深大东路斋区侧台阶\n"
              << u8"G：立德门　Shift+G：立功门　X：星空广场　B：北馆入口　V：石头坞　N：南秋馆泳池　M：汇文楼入口\n"
              << u8"C：周南桥桥下　Shift+C：南区道路　U：周南桥下层　P：召南桥桥面　Y：汇星楼北侧\n"
              << u8"Shift+P：立言门　Shift+U：致远门　Shift+V：南区大草坪\n"
              << u8"Shift+5：致知楼和理工楼 L4　Shift+6：致知楼庭院\n"
              << u8"Shift+7：致理楼 L1 全景　Shift+8：L1 入口　Shift+9：L1–L3 高架连桥\n"
              << u8"Shift+0：致工楼全景　Shift+1：致工楼入口　Shift+2：南区饭堂连桥入口　Shift+3：南区饭堂南侧\n"
              << u8"Shift+4：北门坡道　Shift+Z：杜鹃山步道　Shift+I：文山湖东侧坡路\n"
              << u8"Ctrl+F3：学活至石头坞坡路　Ctrl+F7：师范楼坡路　Ctrl+F8：北夏馆坡路\n"
              << u8"Ctrl+F9：立德门至时光广场　Ctrl+F10：时光广场至立德门　Ctrl+F11：南区宿舍道路\n"
              << u8"Ctrl+Alt+F10：切换立德门演示栏杆\n"
              << u8"Ctrl+F12：风槐斋至深大东路\n"
              << u8"Ctrl+F1：田径场与汇研楼之间\n"
              << u8"Ctrl+F2：汇文楼附近支路　Ctrl+F5：教育学部北侧道路\n"
              << u8"Ctrl+Page Up：元平体育馆后侧环路\n"
              << u8"Ctrl+Page Down：汇德楼B座外围弯路\n"
              << u8"Shift+H：森林校巴花园　Shift+K：阁区商铺坡路　Shift+X：乔相阁对面商铺　Shift+B：顺丰驿站　Shift+Y：修车铺路口\n"
              << u8"Shift+M：Manner 店面　Shift+F：7-11 店面\n"
              << u8"Shift+O：汇元楼附近台阶　Shift+N：荔园小巴初始位置\n"
              << u8"Home：周南桥下层入口　End：周南桥上层道路　Ctrl+U：斋区端楼梯　Ctrl+Shift+U：楼梯桥内侧\n"
              << u8"O：上文山湖西岸　I：上文山湖东南岸\n"
              << u8"Z：汇智楼（图书馆南馆）\n"
              << u8"K：致艺楼全景　Ctrl+K：致理—致艺中庭　F：致艺楼连桥下　[：致理楼　]：演播厅及高架平台\n"
              << u8"Shift+[：古石今人正面　Shift+]：石刻背面\n"
              << u8"\\：致理楼前的高架台阶\n"
              << u8".：汇萃楼全景　,：汇萃楼北侧入口　/：汇意楼及扩建区域\n"
              << u8";：北馆及两侧连廊　'：汇紫楼全景　=：汇紫楼近景　-：汇元楼入口\n"
              << u8"`：演会中心入口　空格：演会中心观众席\n"
              << u8"Page Up：元平体育馆正面　Page Down：北夏馆泳池\n"
              << u8"Insert：高尔夫练习场全景　Delete：击球位　Tab：汇研楼　Backspace：田径场内景\n"
              << u8"T：切换阴影　R：复位　H：说明　Esc：退出\n";
}

void keyCallback(GLFWwindow* window, int key, int, int action, int mods) {
    if (action != GLFW_PRESS) return;
    State& s = *static_cast<State*>(glfwGetWindowUserPointer(window));
    if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(window, GLFW_TRUE);
    if (key == GLFW_KEY_F1) resetView(s,(mods & GLFW_MOD_CONTROL) ? 106 : 1);
    if (key == GLFW_KEY_F2) resetView(s,(mods & GLFW_MOD_CONTROL) ? 107 : 2);
    if (key == GLFW_KEY_F3) resetView(s,(mods & GLFW_MOD_CONTROL) ? 99 : (mods & GLFW_MOD_SHIFT) ? 97 : 3);
    if (key == GLFW_KEY_F4) {
        if (mods & GLFW_MOD_CONTROL) {
            s.kaifengDoorOpen=!s.kaifengDoorOpen;
            std::cout << (s.kaifengDoorOpen ? u8"凯风餐厅入口门打开。\n" : u8"凯风餐厅入口门关闭。\n");
        } else resetView(s,(mods & GLFW_MOD_SHIFT) ? 96 : 4);
    }
    if (key == GLFW_KEY_F5) resetView(s,(mods & GLFW_MOD_CONTROL) ? 108 : 5);
    if (key == GLFW_KEY_F6) {
        if (mods & GLFW_MOD_CONTROL) {
            s.swansActive=!s.swansActive;
            std::cout << (s.swansActive ? u8"黑天鹅继续游动。\n" : u8"黑天鹅游动已暂停。\n");
        } else resetView(s,(mods & GLFW_MOD_SHIFT) ? 95 : 6);
    }
    if (key == GLFW_KEY_F7) resetView(s,(mods & GLFW_MOD_CONTROL) ? 101 : 7);
    if (key == GLFW_KEY_F8) resetView(s,(mods & GLFW_MOD_CONTROL) ? 100 : 8);
    if (key == GLFW_KEY_F9) resetView(s,(mods & GLFW_MOD_CONTROL) ? 102 : 9);
    if (key == GLFW_KEY_F10) {
        if ((mods & (GLFW_MOD_CONTROL | GLFW_MOD_ALT)) == (GLFW_MOD_CONTROL | GLFW_MOD_ALT)) {
            if (s.lideAvailable) {
                s.lideLowered=!s.lideLowered;
                s.lideAngle=s.lideLowered ? s.lideMaxAngle : 0.0f;
                std::cout << (s.lideLowered ? u8"立德门演示栏杆下降。\n" : u8"立德门演示栏杆抬起。\n");
            }
        } else resetView(s,(mods & GLFW_MOD_CONTROL) ? 103 : 10);
    }
    if (key == GLFW_KEY_F11) resetView(s,(mods & GLFW_MOD_CONTROL) ? 104 : 11);
    if (key == GLFW_KEY_F12) resetView(s,(mods & GLFW_MOD_CONTROL) ? 105 : 12);
    if (key == GLFW_KEY_1) resetView(s,(mods & GLFW_MOD_CONTROL) ? 112 : (mods & GLFW_MOD_SHIFT) ? 78 : 13);
    if (key == GLFW_KEY_2) resetView(s,(mods & GLFW_MOD_CONTROL) ? 113 : (mods & GLFW_MOD_SHIFT) ? 79 : 14);
    if (key == GLFW_KEY_3) resetView(s,(mods & GLFW_MOD_CONTROL) ? 114 : (mods & GLFW_MOD_SHIFT) ? 80 : 15);
    if (key == GLFW_KEY_4) resetView(s,(mods & GLFW_MOD_CONTROL) ? 115 : (mods & GLFW_MOD_SHIFT) ? 81 : 16);
    if (key == GLFW_KEY_5) resetView(s,(mods & GLFW_MOD_CONTROL) ? 116 : (mods & GLFW_MOD_SHIFT) ? 72 : 17);
    if (key == GLFW_KEY_6) resetView(s,(mods & GLFW_MOD_CONTROL) ? 117 : (mods & GLFW_MOD_SHIFT) ? 73 : 18);
    if (key == GLFW_KEY_7) resetView(s,(mods & GLFW_MOD_CONTROL) ? 118 : (mods & GLFW_MOD_SHIFT) ? 74 : 19);
    if (key == GLFW_KEY_8) resetView(s,(mods & GLFW_MOD_SHIFT) ? 75 : 20);
    if (key == GLFW_KEY_9) resetView(s,(mods & GLFW_MOD_CONTROL) ? ZHAI_ROADSIDE_STAIR_VIEW : (mods & GLFW_MOD_SHIFT) ? 76 : 21);
    if (key == GLFW_KEY_0) resetView(s,(mods & GLFW_MOD_CONTROL) ? NEW_ZHAI_ARCADE_VIEW : (mods & GLFW_MOD_SHIFT) ? 77 : 22);
    if (key == GLFW_KEY_G) {
        if (mods & GLFW_MOD_CONTROL) selectGateView(s,s.gateSelection+((mods & GLFW_MOD_SHIFT) ? -1 : 1));
        else resetView(s,(mods & GLFW_MOD_SHIFT) ? 67 : 23);
    }
    if (key == GLFW_KEY_X) resetView(s,(mods & GLFW_MOD_SHIFT) ? 86 : 24);
    if (key == GLFW_KEY_B) resetView(s,(mods & GLFW_MOD_SHIFT) ? 87 : 25);
    if (key == GLFW_KEY_V) resetView(s,(mods & GLFW_MOD_SHIFT) ? 71 : 26);
    if (key == GLFW_KEY_N) {
        if (mods & GLFW_MOD_CONTROL) s.placeNames=!s.placeNames;
        else resetView(s,(mods & GLFW_MOD_SHIFT) ? 92 : 27);
    }
    if (key == GLFW_KEY_M) resetView(s,(mods & GLFW_MOD_SHIFT) ? 89 : 28);
    if (key == GLFW_KEY_C) resetView(s,(mods & GLFW_MOD_SHIFT) ? 98 : 29);
    if (key == GLFW_KEY_P) resetView(s,(mods & GLFW_MOD_SHIFT) ? 69 : 30);
    if (key == GLFW_KEY_U) {
        const bool shifted = (mods & GLFW_MOD_SHIFT) != 0;
        resetView(s,(mods & GLFW_MOD_CONTROL) ? (shifted ? ZHOUNAN_ZHAI_INSIDE_VIEW : ZHOUNAN_ZHAI_VIEW)
                                               : (shifted ? 70 : 31));
    }
    if (key == GLFW_KEY_Y) resetView(s,(mods & GLFW_MOD_SHIFT) ? 88 : 32);
    if (key == GLFW_KEY_O) resetView(s,(mods & GLFW_MOD_SHIFT) ? 91 : 33);
    if (key == GLFW_KEY_I) resetView(s,(mods & GLFW_MOD_SHIFT) ? 83 : 34);
    if (key == GLFW_KEY_Z) resetView(s,(mods & GLFW_MOD_SHIFT) ? 82 : 35);
    if (key == GLFW_KEY_K) resetView(s,(mods & GLFW_MOD_CONTROL) ? SOUTH_SPINE_VIEW : (mods & GLFW_MOD_SHIFT) ? 85 : 36);
    if (key == GLFW_KEY_F) resetView(s,(mods & GLFW_MOD_SHIFT) ? 90 : 38);
    if (key == GLFW_KEY_RIGHT_BRACKET) resetView(s,(mods & GLFW_MOD_SHIFT) ? 94 : 43);
    if (key == GLFW_KEY_LEFT_BRACKET) resetView(s,(mods & GLFW_MOD_SHIFT) ? 93 : 42);
    if (key == GLFW_KEY_BACKSLASH) resetView(s,44);
    if (key == GLFW_KEY_PERIOD) resetView(s,46);
    if (key == GLFW_KEY_COMMA) resetView(s,48);
    if (key == GLFW_KEY_SLASH) resetView(s,49);
    if (key == GLFW_KEY_SEMICOLON) resetView(s,50);
    if (key == GLFW_KEY_APOSTROPHE) resetView(s,53);
    if (key == GLFW_KEY_EQUAL) resetView(s,54);
    if (key == GLFW_KEY_MINUS) resetView(s,55);
    if (key == GLFW_KEY_GRAVE_ACCENT) resetView(s,57);
    if (key == GLFW_KEY_SPACE) resetView(s,58);
    if (key == GLFW_KEY_HOME) resetView(s,59);
    if (key == GLFW_KEY_END) resetView(s,60);
    if (key == GLFW_KEY_PAGE_UP) resetView(s,(mods & GLFW_MOD_CONTROL) ? 109 : 61);
    if (key == GLFW_KEY_PAGE_DOWN) resetView(s,(mods & GLFW_MOD_CONTROL) ? 110 : 62);
    if (key == GLFW_KEY_INSERT) resetView(s,63);
    if (key == GLFW_KEY_DELETE) resetView(s,64);
    if (key == GLFW_KEY_TAB) resetView(s,65);
    if (key == GLFW_KEY_BACKSPACE) resetView(s,66);
    if (key == GLFW_KEY_T) {
        if (mods & GLFW_MOD_SHIFT) {
            s.groundFollow = !s.groundFollow;
            s.busFollow = false;
            s.swanFollow = false;
            std::cout << (s.groundFollow ? u8"贴地步行已开启；按 Q/E 恢复自由飞行。\n" : u8"已恢复自由飞行。\n");
        } else s.shadows = !s.shadows;
    }
    if (key == GLFW_KEY_J && (mods & GLFW_MOD_SHIFT)) {
        s.busFollow = !s.busFollow;
        s.groundFollow = false;
        s.swanFollow = false;
        std::cout << (s.busFollow ? u8"跟随小巴已开启；移动或拖动鼠标退出。\n" : u8"已退出小巴跟随视角。\n");
    } else if (key == GLFW_KEY_L && (mods & GLFW_MOD_SHIFT)) {
        s.busAuto = !s.busAuto;
        std::cout << (s.busAuto ? u8"小巴自动前进已开启。\n" : u8"小巴自动前进已暂停。\n");
    } else if (key == GLFW_KEY_J || key == GLFW_KEY_L) s.busAuto=false;
    if (key == GLFW_KEY_W || key == GLFW_KEY_A || key == GLFW_KEY_S || key == GLFW_KEY_D ||
        key == GLFW_KEY_Q || key == GLFW_KEY_E) { s.busFollow=false; s.swanFollow=false; }
    if (s.groundFollow && (key == GLFW_KEY_Q || key == GLFW_KEY_E)) {
        s.groundFollow = false;
        std::cout << u8"已恢复自由飞行。\n";
    }
    if (key == GLFW_KEY_R) {
        const bool lideAvailable=s.lideAvailable; const float lideMaxAngle=s.lideMaxAngle;
        s = State(); s.lideAvailable=lideAvailable; s.lideMaxAngle=lideMaxAngle; resetView(s,1);
    }
    if (key == GLFW_KEY_H) { if (mods & GLFW_MOD_SHIFT) resetView(s,84); else printHelp(); }
}
void mouseButton(GLFWwindow* window, int button, int action, int) {
    State& s = *static_cast<State*>(glfwGetWindowUserPointer(window));
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        s.dragging = action == GLFW_PRESS;
        if (s.dragging) { s.busFollow=false; s.swanFollow=false; }
        glfwGetCursorPos(window, &s.lastX, &s.lastY);
    }
}
void cursorPosition(GLFWwindow* window, double x, double y) {
    State& s = *static_cast<State*>(glfwGetWindowUserPointer(window));
    if (s.dragging) {
        s.yaw += static_cast<float>(x - s.lastX) * .22f;
        s.pitch = glm::clamp(s.pitch - static_cast<float>(y - s.lastY) * .22f, -88.0f, 88.0f);
    }
    s.lastX = x; s.lastY = y;
}
void scrollCallback(GLFWwindow* window, double, double y) {
    State& s = *static_cast<State*>(glfwGetWindowUserPointer(window));
    s.speed = glm::clamp(s.speed * (y > 0 ? 1.25f : .8f), 3.0f, 450.0f);
}

// 与地形网格使用同一份采样值；用于约束自由相机和放置地标。
class TerrainHeights {
public:
    struct Patch {
        std::uint32_t nx,nz;
        float x0,x1,z0,z1;
        std::vector<float> values;
        float at(float x,float z) const {
            float u=glm::clamp((x-x0)/(x1-x0),0.0f,1.0f)*(nx-1);
            float v=glm::clamp((z-z0)/(z1-z0),0.0f,1.0f)*(nz-1);
            unsigned i=std::min(static_cast<unsigned>(u),nx-2),j=std::min(static_cast<unsigned>(v),nz-2);
            float a=u-i,b=v-j,p=values[j*nx+i],q=values[j*nx+i+1];
            float r=values[(j+1)*nx+i],s=values[(j+1)*nx+i+1];
            return b>=a ? (1-b)*p+(b-a)*r+a*s : (1-a)*p+b*s+(a-b)*q;
        }
    };
    std::uint32_t nx, nz;
    float x0, x1, z0, z1;
    std::vector<float> values;
    std::vector<Patch> patches;
    TerrainHeights() : nx(0), nz(0), x0(0), x1(0), z0(0), z1(0) {
        std::ifstream f("assets/campus-heights.bin",std::ios::binary);
        f.read(reinterpret_cast<char*>(&nx),4); f.read(reinterpret_cast<char*>(&nz),4);
        f.read(reinterpret_cast<char*>(&x0),4); f.read(reinterpret_cast<char*>(&x1),4);
        f.read(reinterpret_cast<char*>(&z0),4); f.read(reinterpret_cast<char*>(&z1),4);
        if (!f || nx < 2 || nz < 2 || nx > 1024 || nz > 1024 || x1 <= x0 || z1 <= z0)
            throw std::runtime_error("Invalid campus terrain heights");
        values.resize(static_cast<size_t>(nx)*nz);
        f.read(reinterpret_cast<char*>(&values[0]), static_cast<std::streamsize>(values.size()*sizeof(float)));
        if (!f) throw std::runtime_error("Truncated campus terrain heights");
        // 补丁与实际细分地形共用采样，贴地行走时不会穿进新路面或悬空。
        std::ifstream detail("assets/campus-terrain-patches.bin",std::ios::binary);
        char magic[4]={0,0,0,0}; std::uint32_t count=0;
        detail.read(magic,4); detail.read(reinterpret_cast<char*>(&count),4);
        if (!detail || std::string(magic,4)!="SZP1" || count>16)
            throw std::runtime_error("Invalid campus terrain detail");
        for (std::uint32_t n=0;n<count;++n) {
            Patch p;
            detail.read(reinterpret_cast<char*>(&p.nx),4); detail.read(reinterpret_cast<char*>(&p.nz),4);
            detail.read(reinterpret_cast<char*>(&p.x0),4); detail.read(reinterpret_cast<char*>(&p.x1),4);
            detail.read(reinterpret_cast<char*>(&p.z0),4); detail.read(reinterpret_cast<char*>(&p.z1),4);
            if (!detail || p.nx<2 || p.nz<2 || p.nx>1024 || p.nz>1024 || p.x1<=p.x0 || p.z1<=p.z0 ||
                p.x0<x0 || p.x1>x1 || p.z0<z0 || p.z1>z1)
                throw std::runtime_error("Invalid campus terrain detail bounds");
            p.values.resize(static_cast<size_t>(p.nx)*p.nz);
            detail.read(reinterpret_cast<char*>(&p.values[0]),static_cast<std::streamsize>(p.values.size()*sizeof(float)));
            if (!detail) throw std::runtime_error("Truncated campus terrain detail");
            patches.push_back(std::move(p));
        }
    }
    float at(float x, float z) const {
        for (const Patch& p : patches)
            if (x>=p.x0 && x<=p.x1 && z>=p.z0 && z<=p.z1) return p.at(x,z);
        float u = glm::clamp((x-x0)/(x1-x0),0.0f,1.0f)*(nx-1);
        float v = glm::clamp((z-z0)/(z1-z0),0.0f,1.0f)*(nz-1);
        unsigned i = std::min(static_cast<unsigned>(u),nx-2), j = std::min(static_cast<unsigned>(v),nz-2);
        float a = u-i, b = v-j;
        float p = values[j*nx+i], q = values[j*nx+i+1];
        float r = values[(j+1)*nx+i], s = values[(j+1)*nx+i+1];
        // 与渲染网格的两片三角形使用相同插值，避免坡面上的相机高度漂移。
        return b>=a ? (1-b)*p+(b-a)*r+a*s : (1-a)*p+b*s+(a-b)*q;
    }
};

struct ShuttlePose {
    glm::vec3 position;
    glm::vec2 tangent;
    float yaw, pitch, roll;
};

// 按地图道路中心线计算演示路线；右侧车道偏移和车辆坡度均为近似。
class ShuttleRoute {
    std::vector<glm::vec2> points;
    std::vector<float> distances;
    glm::vec2 center(float distance) const {
        float d=glm::clamp(distance,0.0f,length());
        for (size_t i=1;i<points.size();++i) {
            if (d<=distances[i]) {
                float t=(d-distances[i-1])/(distances[i]-distances[i-1]);
                return glm::mix(points[i-1],points[i],t);
            }
        }
        return points.back();
    }
    glm::vec2 lane(float distance) const {
        float d=glm::clamp(distance,0.0f,length());
        glm::vec2 p=center(d);
        glm::vec2 before=center(d-1.5f), after=center(d+1.5f);
        glm::vec2 direction=glm::normalize(after-before);
        return p+glm::vec2(-direction.y,direction.x)*1.2f;
    }
public:
    ShuttleRoute() {
        std::ifstream file("assets/campus-shuttle-route.bin",std::ios::binary);
        char magic[4]={0,0,0,0}; std::uint32_t count=0;
        file.read(magic,4); file.read(reinterpret_cast<char*>(&count),4);
        if (!file || magic[0]!='S' || magic[1]!='Z' || magic[2]!='R' || magic[3]!='1' || count<2 || count>10000)
            throw std::runtime_error("Invalid campus shuttle route");
        for (std::uint32_t i=0;i<count;++i) {
            float x=0,z=0;
            file.read(reinterpret_cast<char*>(&x),4); file.read(reinterpret_cast<char*>(&z),4);
            if (!file || !std::isfinite(x) || !std::isfinite(z)) throw std::runtime_error("Truncated campus shuttle route");
            glm::vec2 p(x,z);
            float step=points.empty() ? 0.0f : glm::length(p-points.back());
            if (!points.empty() && step<.01f) throw std::runtime_error("Duplicate campus shuttle route point");
            points.push_back(p);
            distances.push_back(distances.empty() ? 0.0f : distances.back()+step);
        }
    }
    float length() const { return distances.back(); }
    ShuttlePose pose(float distance, const TerrainHeights& heights) const {
        glm::vec2 p=lane(distance), a=lane(distance-1.5f), b=lane(distance+1.5f);
        glm::vec2 direction=glm::normalize(b-a);
        glm::vec2 right(-direction.y,direction.x);
        float wheel[2][2];
        for (int i=0;i<2;++i) for (int j=0;j<2;++j) {
            glm::vec2 contact=p+direction*((i ? 1.0f : -1.0f)*1.61f)+right*((j ? 1.0f : -1.0f)*.83f);
            wheel[i][j]=heights.at(contact.x,contact.y);
        }
        // 四个接地点拟合车身平面；坡地上的车头和两侧随路面倾斜。
        float level=(wheel[0][0]+wheel[0][1]+wheel[1][0]+wheel[1][1])*.25f;
        float longitudinal=(wheel[1][0]+wheel[1][1]-wheel[0][0]-wheel[0][1])/(4*1.61f);
        float cross=(wheel[0][1]+wheel[1][1]-wheel[0][0]-wheel[1][0])/(4*.83f);
        float pitch=std::atan(longitudinal);
        ShuttlePose result={{p.x,level+.035f,p.y},direction,
                            -glm::degrees(std::atan2(direction.y,direction.x)),
                            glm::degrees(pitch),-glm::degrees(std::atan(cross/std::cos(pitch)))};
        return result;
    }
};

struct SwanPose {
    glm::vec3 position;
    glm::vec2 tangent;
    float yaw;
};

// 活动轨迹由湖区资源生成时约束在水域内部；不声称为真实动物的固定路线。
class SwanRoute {
    float centerX,centerZ,radiusX,radiusZ,level,speed,phases[5];
public:
    SwanRoute() : centerX(0),centerZ(0),radiusX(0),radiusZ(0),level(0),speed(0),phases{} {
        std::ifstream file("assets/campus-wenshan-swan-route.bin",std::ios::binary);
        char magic[4]={0,0,0,0}; float values[11]={};
        file.read(magic,4); file.read(reinterpret_cast<char*>(values),sizeof(values));
        if (!file || magic[0]!='S' || magic[1]!='W' || magic[2]!='N' || magic[3]!='1')
            throw std::runtime_error("Invalid Wenshan swan route");
        for (float v:values) if (!std::isfinite(v)) throw std::runtime_error("Invalid Wenshan swan route value");
        centerX=values[0]; centerZ=values[1]; radiusX=values[2]; radiusZ=values[3]; level=values[4]; speed=values[5];
        for (int i=0;i<5;++i) phases[i]=values[6+i];
        if (radiusX<=2 || radiusZ<=2 || speed<=0 || speed>=.2f) throw std::runtime_error("Invalid Wenshan swan route size");
    }
    SwanPose pose(float elapsed,int index) const {
        float a=std::fmod(elapsed*speed+phases[index],2.0f*PI);
        glm::vec2 tangent=glm::normalize(glm::vec2(-radiusX*std::sin(a),radiusZ*std::cos(a)));
        const float spread[5]={0.0f,-.22f,.28f,-.18f,0.0f};
        glm::vec2 across(-tangent.y,tangent.x);
        SwanPose result={{centerX+radiusX*std::cos(a)+spread[index]*across.x,level,
                          centerZ+radiusZ*std::sin(a)+spread[index]*across.y},tangent,
                         -glm::degrees(std::atan2(tangent.y,tangent.x))};
        return result;
    }
};

// 两扇入口门共享网格；铰链坐标与墙的朝向由独立模型一起导出。
class KaifengEntry {
    glm::vec2 left,right,tangent;
    float floorY;
public:
    KaifengEntry() : left(0),right(0),tangent(1,0),floorY(0) {
        std::ifstream file("assets/campus-kaifeng-entry.bin",std::ios::binary);
        char magic[4]={0,0,0,0}; float value[7]={};
        file.read(magic,4); file.read(reinterpret_cast<char*>(value),sizeof(value));
        if (!file || magic[0]!='K' || magic[1]!='F' || magic[2]!='E' || magic[3]!='1')
            throw std::runtime_error("Invalid Kai Feng entrance resource");
        for (float v:value) if (!std::isfinite(v)) throw std::runtime_error("Invalid Kai Feng entrance value");
        left=glm::vec2(value[0],value[1]); right=glm::vec2(value[2],value[3]);
        tangent=glm::vec2(value[4],value[5]); floorY=value[6];
        if (std::abs(glm::length(tangent)-1.0f)>.01f || glm::length(right-left)<2.0f)
            throw std::runtime_error("Invalid Kai Feng door dimensions");
    }
    glm::mat4 leaf(const glm::mat4& root, bool rightSide, float opening) const {
        glm::vec2 hinge=rightSide ? right : left;
        glm::mat4 basis(1.0f);
        basis[0]=glm::vec4(tangent.x,0,tangent.y,0);
        basis[1]=glm::vec4(0,1,0,0);
        basis[2]=glm::vec4(-tangent.y,0,tangent.x,0);
        basis[3]=glm::vec4(hinge.x,floorY,hinge.y,1);
        return glm::rotate(root*basis,glm::radians(rightSide ? 180.0f+75.0f*opening : -75.0f*opening),glm::vec3(0,1,0));
    }
};

// 保存可以行走的桥板上表面，步行时选择脚下可抵达的一层，避免从地面跳到高架桥。
class WalkSurfaces {
    struct Face {
        glm::vec3 a,b,c;
        float minX,maxX,minZ,maxZ;
    };
    std::vector<Face> faces;
    void add(const char* path) {
        std::vector<Vertex> vertices=readMesh(path);
        for (size_t i=0;i+2<vertices.size();i+=3) {
            const Vertex &a=vertices[i], &b=vertices[i+1], &c=vertices[i+2];
            if (std::min(a.n.y,std::min(b.n.y,c.n.y))<.7f) continue;
            float area=(b.p.z-c.p.z)*(a.p.x-c.p.x)+(c.p.x-b.p.x)*(a.p.z-c.p.z);
            if (std::abs(area)<.00001f) continue;
            Face f={a.p,b.p,c.p,std::min(a.p.x,std::min(b.p.x,c.p.x)),
                    std::max(a.p.x,std::max(b.p.x,c.p.x)),std::min(a.p.z,std::min(b.p.z,c.p.z)),
                    std::max(a.p.z,std::max(b.p.z,c.p.z))};
            faces.push_back(f);
        }
    }
public:
    WalkSurfaces() {
        add("assets/campus-bridge.mesh");
        add("assets/campus-zhounan-lower.mesh");
        add("assets/campus-zhounan-zhai-treads.mesh");
        add("assets/campus-zhai-paving.mesh");
        add("assets/campus-zhiyi-skywalk.mesh");
        add("assets/campus-zhili-high-walkways.mesh");
        add("assets/campus-network-stairs.mesh");
        // 元平入口与二层外梯：只加入踏面/平台，不把屋盖当作可行走层。
        add("assets/campus-yuanping-entry-walk.mesh");
        add("assets/campus-stadium-entry-walk.mesh");
        add("assets/campus-stadium-surface.mesh");
        add("assets/campus-north-library-entry-walk.mesh");
        add("assets/campus-north-library-upper-walk.mesh");
        add("assets/campus-lide-corridor-paving.mesh");
        add("assets/campus-kaifeng-stairs.mesh");
        add("assets/campus-kaifeng-apron.mesh");
        add("assets/campus-kaifeng-interior.mesh");
        for (const char* index:{"assets/campus-gates.tsv","assets/campus-qiaoyuan-dorms.tsv","assets/campus-south-dorms.tsv",
                                "assets/campus-south-spine.tsv"}) {
            std::ifstream file(index);std::string line;
            if (!file) throw std::runtime_error(std::string("Missing walkable model index: ")+index);
            while (std::getline(file,line)) {
                std::istringstream row(line);std::string tag,id,mesh;
                if (row>>tag>>id>>mesh && tag=="W") add(("assets/"+mesh).c_str());
            }
        }
    }
    float at(float x, float z, float previousFoot, float ground) const {
        const float limit=previousFoot+.8f;
        float best=ground;
        bool found=ground<=limit;
        for (size_t i=0;i<faces.size();++i) {
            const Face& f=faces[i];
            if (x<f.minX-.001f || x>f.maxX+.001f || z<f.minZ-.001f || z>f.maxZ+.001f) continue;
            float denominator=(f.b.z-f.c.z)*(f.a.x-f.c.x)+(f.c.x-f.b.x)*(f.a.z-f.c.z);
            float u=((f.b.z-f.c.z)*(x-f.c.x)+(f.c.x-f.b.x)*(z-f.c.z))/denominator;
            float v=((f.c.z-f.a.z)*(x-f.c.x)+(f.a.x-f.c.x)*(z-f.c.z))/denominator;
            float w=1-u-v;
            if (std::min(u,std::min(v,w))<-.0001f) continue;
            float y=u*f.a.y+v*f.b.y+w*f.c.y;
            if (y<=limit && y>=ground-.05f && (!found || y>best)) { best=y; found=true; }
        }
        return found ? best : ground;
    }
};

// 使用实际相机的高度选择逻辑，逐点核验元平的上、下楼路线。
int verifyEntryWalk(const char* filename,size_t minimumRoutes,int groundMode) {
    std::ifstream input(filename);
    if (!input) throw std::runtime_error("Missing entry routes");
    std::map<std::string,std::vector<glm::vec3> > routes;
    std::string line;
    while (std::getline(input,line)) {
        if (line.empty() || line[0]=='#') continue;
        std::istringstream row(line); std::string tag,id; glm::vec3 p;
        if (!(row>>tag>>id>>p.x>>p.y>>p.z) || tag!="P" ||
            !std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
            throw std::runtime_error("Invalid Yuanping entry route");
        for (char c:id) if (!(std::isalnum(static_cast<unsigned char>(c)) || c=='_'))
            throw std::runtime_error("Invalid entry route ID");
        routes[id].push_back(p);
    }
    if (routes.size()<minimumRoutes) throw std::runtime_error("Incomplete entry routes");
    TerrainHeights terrain; WalkSurfaces walking;
    bool passed=true; std::ostringstream report;
    report << std::setprecision(9) << "{\"routes\":["; bool first=true;
    for (const auto& route:routes) {
        if (route.second.size()<2) throw std::runtime_error("Entry route has too few samples");
        if (!first) report << ','; first=false;
        report << "{\"id\":\"" << route.first << "\",\"samples\":" << route.second.size() << ",\"directions\":[";
        for (int reverse=0;reverse<2;++reverse) {
            if (reverse) report << ',';
            const glm::vec3 start=reverse ? route.second.back() : route.second.front();
            const float startGround=terrain.at(start.x,start.z);
            const bool groundedStart=groundMode==1 || (groundMode==2 && !reverse);
            float previous=groundedStart ? startGround : start.y;
            if (groundedStart && std::abs(start.y-startGround)>.12f)
                throw std::runtime_error("Entry route endpoint is not grounded");
            float maxError=0,maxRise=0,maxDrop=0; size_t failures=0,firstFailure=0;
            for (size_t i=0;i<route.second.size();++i) {
                const glm::vec3& p=route.second[reverse ? route.second.size()-1-i : i];
                const float actual=walking.at(p.x,p.z,previous,terrain.at(p.x,p.z));
                const float error=std::abs(actual-p.y);
                // 5 mm 容差小于铺面离地量，避免仅靠地形兜底也被误判为踏面匹配。
                if (error>.005f || std::abs(actual-previous)>.31f) { if (!failures) firstFailure=i; ++failures; }
                maxError=std::max(maxError,error);
                maxRise=std::max(maxRise,actual-previous);maxDrop=std::max(maxDrop,previous-actual);
                previous=actual;
            }
            passed=passed && failures==0;
            report << "{\"reverse\":" << (reverse ? "true" : "false") << ",\"max_foot_error_m\":" << maxError
                   << ",\"max_rise_m\":" << maxRise << ",\"max_drop_m\":" << maxDrop
                   << ",\"failures\":" << failures << ",\"first_failure_sample\":" << firstFailure << '}';
        }
        report << "]}";
    }
    report << "],\"status\":\"" << (passed ? "passed" : "failed")
           << "\",\"ground_mode\":" << groundMode
           << ",\"tolerance_m\":0.005,\"max_allowed_step_m\":0.31}\n";
    std::cout << report.str(); return passed ? 0 : 1;
}

enum Texture { WHITE, CONCRETE, STONE, PAVING, GRASS, GLASS, WATER, HUIDIAN, HUIZHI,
    TREE_BARK, TREE_LEAVES, PALM_ATLAS, TREE_CANOPY, TREE_CLUSTER, CAMPUS_GROUND, FACADE_CREAM, FACADE_RED, FACADE_CITY,
    FACADE_STRIP, FACADE_PERFORATED, FACADE_TENG, FACADE_ATRIUM, ARCHIVE_SEAL,
    BASKETBALL_COURT, TENNIS_COURT, SUNDIAL_FACE, SUNDIAL_STONE, FACADE_QIAOMU, FACADE_QIAOSEN, FACADE_ZHAI,
    FACADE_GATE, GATE_SIGN, GATE_ADDRESS, SPORTS_FENCE, FACADE_FORTRESS, STONE_PLAZA, SOUTH_POOL, FACADE_ACTIVITY, FACADE_HUIWEN,
    FACADE_HUIWEN_RED, FACADE_HUIXING, FACADE_HUIXING_CORE, FACADE_HUIXING_PODIUM, HUIXING_ROOF,
    FACADE_HUIZHI, HUIZHI_ENTRANCE, FACADE_ZHIYI_VENT, FACADE_ZHIYI_PODIUM,
    STUDIO_SIGN, FACADE_ZHILI_SIDE, ZHILI_A_SIGN, ZHIGONG_LOUVER, ZHIGONG_SIGN, HUIYUAN_SIGN, HUIYUAN_MOTTO, YUANPING_RELIEF, YUANPING_SEAL, YUANPING_NAME, GOLF_DISTANCE, NORTH_CITY_GROUND, HUIZI_ROOF, HUIZI_EQUIPMENT, FACADE_HUIYAN, STADIUM_SURFACE, STADIUM_SEAT, ZHIYUAN_WORDS, ZHIYUAN_BASE, CANTEEN_SIGN, DUJUAN_BOARDWALK, CAMPUS_ASPHALT,
    QIAOYUAN_GARDEN_WALK, QIAOYUAN_BRICK, QIAOYUAN_GARDEN_WORDS, XIYUAN_SIGNS, XIYUAN_FACADE, XIYUAN_PAVING, XIYUAN_WINDOW, MANNER_WALL, CAMPUS_SIDEWALK, SHUTTLE_SIGN, WENSHAN_ANCIENT_STONE, SWAN_FEATHERS, KAIFENG_TILE, ZHOUNAN_ZHAI_TILE, YUANPING_ENTRY_PAVING, STADIUM_ENTRY_PAVING, NORTH_LIBRARY_ENTRY_PAVING, TEXTURE_COUNT };
const char* textureFiles[TREE_BARK] = {"white", "concrete", "stone", "paving", "grass", "glass", "water", "huidian", "huizhi"};

class Renderer {
public:
    Mesh cube, cylinder, sphere, torus, heart, signPlane, huixingFrame, northLibraryWindows, treeBark, treeLeaves, nearLeafCards, palmMesh, royalPalmLeaves, farTrunk, farCrown, farLeafCards;
    Mesh terrain, northCityGround, campusWater, waterBank, wenshanTerraces, wenshanPaving, wenshanAncientStone, wenshanAncientStonePlanting, wallCream, wallRed, wallCity, wallHuiyan, wallStrip, wallPerforated, wallTeng, landmarkGlass, roofs;
    Mesh canteenGlass, canteenFrame, canteenRoof, canteenSign;
    Mesh roadSurface, roadMarkings, roadCurbs, dujuanTrail, dujuanBoardwalk;
    Mesh networkSidewalk, networkPaths, networkShrubs, networkStairs, lideWalk, lideWalkEdge, annotatedShrubs;
    Mesh corridorPaving, corridorLawn, corridorBed, corridorCurbs, corridorLeaves;
    Mesh networkVerges, networkVergeGrass, networkLamps, networkLampHeads;
    Mesh shuttleBody, shuttleMetal, shuttleInterior, shuttleSeats, shuttleGlass, shuttleLamps, shuttleSign;
    Mesh swanAdult, swanFeather, swanBill, swanBillTip, swanAdultEye, swanChick, swanChickWing, swanChickBill, swanChickEye, swanWake;
    Mesh kaifengShell, kaifengRoof, kaifengWindows, kaifengYellow, kaifengRed, kaifengFrame, kaifengRail;
    Mesh kaifengStairs, kaifengApron, kaifengInterior, kaifengFurniture, kaifengDoorGlass, kaifengDoorFrame;
    Mesh gardenWalk, gardenBrick, gardenMulch, gardenFrame, gardenWords, gardenWood, gardenWoodTops, gardenPlants, gardenPlantsLight, gardenStones;
    Mesh xiyuanWall, xiyuanFrame, xiyuanRailing, shopGlass, shopFrames, shopSigns, shopCanopy, shopShell, shopRoof, shopPaving, shopBicycle;
    Mesh shopOppositePaving, shopMembrane, shopAwningFrame, shopBaseDark, shopBaseBrick, shopRedPaving, shopPlantBed, shopPlantLeaves, shopMannerWall;
    Mesh archiveWall, archiveSeal, archiveFins, archiveDoor;
    Mesh bridge, bridgeRails, bridgeRoof, bridgeSupport, bridgeRoad, bridgeMarkings, bridgeLower, bridgeTruss, bridgeGlass;
    Mesh zhounanYellow, zhounanPots, zhounanPlants, zhounanCeiling;
    Mesh zhounanZhaiTreads, zhounanZhaiWalls;
    Mesh redStands, blueStands, orangeStands, yellowStands, standBase, standLines, stadiumCanopy, stadiumMasts, stadiumLamps;
    Mesh stadiumSurface, stadiumGoals, stadiumEntryPaving, stadiumEntryEdges, stadiumEntryMetal;
    Mesh northEntrySteps,northEntryPaving,northEntryEdges,northUpperDeck,northUpperStairs,northUpperRails,northEntryFrames;
    Mesh sportsShell, sportsRoof, basketballCourt, tennisCourt, sportsRails, sportsNetting, basketballBoards, basketballRims;
    Mesh wallFortress, wallActivity, activityFoundation, teachersFoundation, northsportsFoundation, southdormFoundation, windzhaiFoundation, huiwenroadFoundation, facultyroadFoundation, wallHuiwen, huiwenDetail, huiwenTerrace, huiwenRed, huiwenSteps;
    Mesh zhiyiShell, zhiyiGlass, zhiyiVents, zhiyiRails, zhiyiPodium, zhiyiSkywalk, zhiyiSkyRails;
    Mesh zhizhenShell, zhizhenGlass, zhizhenSide, zhizhenFrame, zhizhenPodium;
    Mesh zhizhiShell, zhizhiGlass, zhizhiFrame, l4Glass, l4Frame, l4Garden, l4Equipment;
    Mesh zhigongShell, zhigongGlass, zhigongDark, zhigongPerforated, zhigongLouvers, zhigongRails, zhigongSteps, zhigongSign;
    Mesh zhigongCarsLight, zhigongCarsDark, zhigongCarsGlass, zhigongCarsWheels, zhigongChargers;
    Mesh studioShell, studioGlass, studioSign, studioColumns;
    Mesh zhiliGlass, zhiliFrame, zhiliSide;
    Mesh zhiliAShell, zhiliAGlass, zhiliAFrame, zhiliASign, zhiliHighWalkways, zhiliHighRails;
    Mesh zhiliCrown, zhiliCrownGlass, zhiliCrownFrame;
    Mesh huicuiShell, huicuiGlass, huicuiUpper, huicuiStage, huicuiFrames, huicuiSkylights;
    Mesh huiziShell, huiziGround, huiziCanopy, huiziPaving, huiziRoof, huiziEquipment;
    Mesh huiyuanShell, huiyuanGround, huiyuanSign, huiyuanMotto, huiyuanPaving, huiyuanPlanting, huiyuanEntryRoof;
    Mesh yanhuiRoof, yanhuiUnder, yanhuiSteel, yanhuiStone, yanhuiFloor, yanhuiSeats, yanhuiSeatDetail, yanhuiStage, yanhuiScreen, yanhuiSteps, yanhuiSeams, yanhuiPineTrunks, yanhuiPineCrowns;
    Mesh huixingSteps, huixingHedges;
    Mesh stoneShell, stoneFlat, stonePlaza, southPool, stoneCanopies, stoneRails;
    Mesh yuanpingShell, yuanpingBase, yuanpingFoundation, yuanpingRoof, yuanpingTruss, yuanpingRelief, yuanpingSeal, yuanpingGlass, yuanpingSteps, yuanpingName, yuanpingGarden, yuanpingEntryPaving;
    Mesh beixiaShell, beixiaRoof, beixiaFrame, beixiaSeats, beixiaDeck, beixiaLanes, beixiaPractice, beixiaCoping, beixiaUnder;
    Mesh golfFloor, golfRoof, golfFrame, golfMats, golfSigns;
    Mesh sundialFace, sundialStone, sundialPointer, timePaving, timeCurb, timeHedges, timeHedgeLeaves;
    Mesh wallZhai, zhaiStone, zhaiPaving, alumniGarden;
    TerrainHeights heights;
    ShuttleRoute shuttleRoute;
    SwanRoute swanRoute;
    KaifengEntry kaifengEntry;
    WalkSurfaces walking;
    std::unique_ptr<CampusAsset> lideAsset;
    CampusModelLayers gates,qiaoyuanDorms,southDorms,southSpine;
    PlaceLabels placeLabels;
    GLuint scene, depth, sky, shadowFbo, shadowTexture, emptyVao, nearInstanceVbo, farInstanceVbo, textures[TEXTURE_COUNT];
    GLsizei nearTreeCount, farTreeCount;
    std::vector<glm::vec4> treePositions;
    bool depthPass;
    glm::mat4 lightSpace;

    Renderer() : cube(cubeGeometry()), cylinder(cylinderGeometry()), sphere(sphereGeometry()),
                 torus(torusGeometry()), heart(heartGeometry()), signPlane(signPlaneGeometry()), huixingFrame(huixingFrameGeometry()), northLibraryWindows(northLibraryWindowGeometry()), treeBark(readMesh("assets/tree-bark.mesh")),
                 treeLeaves(distantTreeCrown(10,7,.70f)), nearLeafCards(treeLeafCards(3)), palmMesh(readMesh("assets/palm.mesh")), royalPalmLeaves(royalPalmCrown()),
                 farTrunk(distantTreeTrunk()), farCrown(distantTreeCrown(6,4)), farLeafCards(treeLeafCards(2)),
                 terrain(readMesh("assets/campus-terrain.mesh")), northCityGround(readMesh("assets/campus-north-city-ground.mesh")), campusWater(readMesh("assets/campus-water.mesh")),
                 waterBank(readMesh("assets/campus-water-bank.mesh")), wenshanTerraces(readMesh("assets/campus-wenshan-terraces.mesh")),
                 wenshanPaving(readMesh("assets/campus-wenshan-paving-detail.mesh")), wenshanAncientStone(readMesh("assets/campus-wenshan-ancient-stone.mesh")),
                 wenshanAncientStonePlanting(readMesh("assets/campus-wenshan-ancient-stone-planting.mesh")),
                 wallCream(readMesh("assets/campus-wall-cream.mesh")), wallRed(readMesh("assets/campus-wall-red.mesh")),
                 wallCity(readMesh("assets/campus-wall-city.mesh")), wallHuiyan(readMesh("assets/campus-wall-huiyan.mesh")),
                 wallStrip(readMesh("assets/campus-wall-strip.mesh")), wallPerforated(readMesh("assets/campus-wall-perforated.mesh")),
                 wallTeng(readMesh("assets/campus-wall-teng.mesh")),
                 landmarkGlass(readMesh("assets/campus-landmark-glass.mesh")),
                 roofs(readMesh("assets/campus-roofs.mesh")),
                 canteenGlass(readMesh("assets/campus-canteen-glass.mesh")), canteenFrame(readMesh("assets/campus-canteen-frame.mesh")),
                 canteenRoof(readMesh("assets/campus-canteen-roof.mesh")), canteenSign(readMesh("assets/campus-canteen-sign.mesh")),
                 roadSurface(readMesh("assets/campus-road-surface.mesh")), roadMarkings(readMesh("assets/campus-road-markings.mesh")),
                 roadCurbs(readMesh("assets/campus-road-curbs.mesh")), dujuanTrail(readMesh("assets/campus-dujuan-trail.mesh")),
                 dujuanBoardwalk(readMesh("assets/campus-dujuan-boardwalk.mesh")),
                 networkSidewalk(readMesh("assets/campus-network-sidewalk.mesh")), networkPaths(readMesh("assets/campus-network-paths.mesh")),
                 networkShrubs(readMesh("assets/campus-network-shrubs.mesh")), networkStairs(readMesh("assets/campus-network-stairs.mesh")),
                 lideWalk(readMesh("assets/campus-lide-walk.mesh")), lideWalkEdge(readMesh("assets/campus-lide-walk-edge.mesh")),
                 annotatedShrubs(readMesh("assets/campus-annotated-shrubs.mesh",true)),
                 corridorPaving(readMesh("assets/campus-lide-corridor-paving.mesh")),
                 corridorLawn(readMesh("assets/campus-lide-corridor-lawn.mesh")),
                 corridorBed(readMesh("assets/campus-lide-corridor-bed.mesh")),
                 corridorCurbs(readMesh("assets/campus-lide-corridor-curbs.mesh")),
                 corridorLeaves(readMesh("assets/campus-lide-corridor-leaves.mesh")),
                 networkVerges(readMesh("assets/campus-network-verges.mesh")), networkVergeGrass(readMesh("assets/campus-network-verge-grass.mesh")),
                 networkLamps(readMesh("assets/campus-network-lamps.mesh")), networkLampHeads(readMesh("assets/campus-network-lamp-heads.mesh")),
                 shuttleBody(readMesh("assets/campus-shuttle-body.mesh")), shuttleMetal(readMesh("assets/campus-shuttle-metal.mesh")),
                 shuttleInterior(readMesh("assets/campus-shuttle-interior.mesh")), shuttleSeats(readMesh("assets/campus-shuttle-seats.mesh")),
                 shuttleGlass(readMesh("assets/campus-shuttle-glass.mesh")), shuttleLamps(readMesh("assets/campus-shuttle-lamps.mesh")),
                 shuttleSign(readMesh("assets/campus-shuttle-sign.mesh")),
                 swanAdult(readMesh("assets/campus-wenshan-swan-adult.mesh")), swanFeather(readMesh("assets/campus-wenshan-swan-feather.mesh")),
                 swanBill(readMesh("assets/campus-wenshan-swan-bill.mesh")), swanBillTip(readMesh("assets/campus-wenshan-swan-bill-tip.mesh")),
                 swanAdultEye(readMesh("assets/campus-wenshan-swan-adult-eye.mesh")),
                 swanChick(readMesh("assets/campus-wenshan-swan-chick.mesh")), swanChickWing(readMesh("assets/campus-wenshan-swan-chick-wing.mesh")),
                 swanChickBill(readMesh("assets/campus-wenshan-swan-chick-bill.mesh")), swanChickEye(readMesh("assets/campus-wenshan-swan-chick-eye.mesh")),
                 swanWake(readMesh("assets/campus-wenshan-swan-wake.mesh")),
                 kaifengShell(readMesh("assets/campus-kaifeng-shell.mesh")), kaifengRoof(readMesh("assets/campus-kaifeng-roof.mesh")),
                 kaifengWindows(readMesh("assets/campus-kaifeng-windows.mesh")), kaifengYellow(readMesh("assets/campus-kaifeng-yellow.mesh")),
                 kaifengRed(readMesh("assets/campus-kaifeng-red.mesh")), kaifengFrame(readMesh("assets/campus-kaifeng-frame.mesh")),
                 kaifengRail(readMesh("assets/campus-kaifeng-rail.mesh")), kaifengStairs(readMesh("assets/campus-kaifeng-stairs.mesh")),
                 kaifengApron(readMesh("assets/campus-kaifeng-apron.mesh")), kaifengInterior(readMesh("assets/campus-kaifeng-interior.mesh")),
                 kaifengFurniture(readMesh("assets/campus-kaifeng-furniture.mesh")), kaifengDoorGlass(readMesh("assets/campus-kaifeng-door-glass.mesh")),
                 kaifengDoorFrame(readMesh("assets/campus-kaifeng-door-frame.mesh")),
                 gardenWalk(readMesh("assets/campus-qiaoyuan-garden-walk.mesh")), gardenBrick(readMesh("assets/campus-qiaoyuan-garden-brick.mesh")),
                 gardenMulch(readMesh("assets/campus-qiaoyuan-garden-mulch.mesh")), gardenFrame(readMesh("assets/campus-qiaoyuan-garden-frame.mesh")),
                 gardenWords(readMesh("assets/campus-qiaoyuan-garden-words.mesh")), gardenWood(readMesh("assets/campus-qiaoyuan-garden-wood.mesh")),
                 gardenWoodTops(readMesh("assets/campus-qiaoyuan-garden-wood-tops.mesh")), gardenPlants(readMesh("assets/campus-qiaoyuan-garden-plants.mesh")),
                 gardenPlantsLight(readMesh("assets/campus-qiaoyuan-garden-plants-light.mesh")), gardenStones(readMesh("assets/campus-qiaoyuan-garden-stones.mesh")),
                 xiyuanWall(readMesh("assets/campus-wall-xiyuan.mesh")), xiyuanFrame(readMesh("assets/campus-xiyuan-frame.mesh")), xiyuanRailing(readMesh("assets/campus-xiyuan-railing.mesh")),
                 shopGlass(readMesh("assets/campus-xiyuan-shops-glass.mesh")), shopFrames(readMesh("assets/campus-xiyuan-shops-frames.mesh")),
                 shopSigns(readMesh("assets/campus-xiyuan-shops-signs.mesh")), shopCanopy(readMesh("assets/campus-xiyuan-shops-canopy.mesh")),
                 shopShell(readMesh("assets/campus-xiyuan-shops-shell.mesh")), shopRoof(readMesh("assets/campus-xiyuan-shops-roof.mesh")),
                 shopPaving(readMesh("assets/campus-xiyuan-shops-paving.mesh")), shopBicycle(readMesh("assets/campus-xiyuan-shops-bicycle.mesh")),
                 shopOppositePaving(readMesh("assets/campus-xiyuan-shops-opposite-paving.mesh")),
                 shopMembrane(readMesh("assets/campus-xiyuan-shops-membrane.mesh")), shopAwningFrame(readMesh("assets/campus-xiyuan-shops-awning-frame.mesh")),
                 shopBaseDark(readMesh("assets/campus-xiyuan-shops-base-dark.mesh")), shopBaseBrick(readMesh("assets/campus-xiyuan-shops-base-brick.mesh")),
                 shopRedPaving(readMesh("assets/campus-xiyuan-shops-red-paving.mesh")), shopPlantBed(readMesh("assets/campus-xiyuan-shops-plant-bed.mesh")),
                 shopPlantLeaves(readMesh("assets/campus-xiyuan-shops-plant-leaves.mesh")), shopMannerWall(readMesh("assets/campus-xiyuan-shops-manner-wall.mesh")),
                 archiveWall(readMesh("assets/campus-archive-wall.mesh")), archiveSeal(readMesh("assets/campus-archive-seal.mesh")),
                 archiveFins(readMesh("assets/campus-archive-fins.mesh")), archiveDoor(readMesh("assets/campus-archive-door.mesh")),
                 bridge(readMesh("assets/campus-bridge.mesh")), bridgeRails(readMesh("assets/campus-bridge-rails.mesh")),
                 bridgeRoof(readMesh("assets/campus-bridge-roof.mesh")), bridgeSupport(readMesh("assets/campus-bridge-support.mesh")),
                 bridgeRoad(readMesh("assets/campus-bridge-road.mesh")), bridgeMarkings(readMesh("assets/campus-bridge-markings.mesh")),
                 bridgeLower(readMesh("assets/campus-zhounan-lower.mesh")),
                 bridgeTruss(readMesh("assets/campus-zhounan-truss.mesh")),
                 bridgeGlass(readMesh("assets/campus-zhaonan-glass.mesh")),
                 zhounanYellow(readMesh("assets/campus-zhounan-yellow.mesh")),
                 zhounanPots(readMesh("assets/campus-zhounan-pots.mesh")),
                 zhounanPlants(readMesh("assets/campus-zhounan-plants.mesh")),
                 zhounanCeiling(readMesh("assets/campus-zhounan-ceiling.mesh")),
                 zhounanZhaiTreads(readMesh("assets/campus-zhounan-zhai-treads.mesh")),
                 zhounanZhaiWalls(readMesh("assets/campus-zhounan-zhai-walls.mesh")),
                 redStands(readMesh("assets/campus-stands-red.mesh")), blueStands(readMesh("assets/campus-stands-blue.mesh")),
                 orangeStands(readMesh("assets/campus-stands-orange.mesh")), yellowStands(readMesh("assets/campus-stands-yellow.mesh")),
                 standBase(readMesh("assets/campus-stands-base.mesh")), standLines(readMesh("assets/campus-stands-lines.mesh")),
                 stadiumCanopy(readMesh("assets/campus-stadium-canopy.mesh")), stadiumMasts(readMesh("assets/campus-stadium-masts.mesh")),
                 stadiumLamps(readMesh("assets/campus-stadium-lamps.mesh")), stadiumSurface(readMesh("assets/campus-stadium-surface.mesh")),
                 stadiumGoals(readMesh("assets/campus-stadium-goals.mesh")),
                 stadiumEntryPaving(readMesh("assets/campus-stadium-entry-paving.mesh")),
                 stadiumEntryEdges(readMesh("assets/campus-stadium-entry-edges.mesh")),
                 stadiumEntryMetal(readMesh("assets/campus-stadium-entry-metal.mesh")),
                 northEntrySteps(readMesh("assets/campus-north-library-entry-steps.mesh")),
                 northEntryPaving(readMesh("assets/campus-north-library-entry-paving.mesh")),
                 northEntryEdges(readMesh("assets/campus-north-library-entry-edges.mesh")),
                 northUpperDeck(readMesh("assets/campus-north-library-upper-deck.mesh")),
                 northUpperStairs(readMesh("assets/campus-north-library-upper-stairs.mesh")),
                 northUpperRails(readMesh("assets/campus-north-library-upper-rails.mesh")),
                 northEntryFrames(readMesh("assets/campus-north-library-upper-frames.mesh")),
                 sportsShell(readMesh("assets/campus-sports-shell.mesh")), sportsRoof(readMesh("assets/campus-sports-roof.mesh")),
                 basketballCourt(readMesh("assets/campus-basketball.mesh")), tennisCourt(readMesh("assets/campus-tennis.mesh")),
                 sportsRails(readMesh("assets/campus-sports-rails.mesh")),
                 sportsNetting(readMesh("assets/campus-sports-netting.mesh")),
                 basketballBoards(readMesh("assets/campus-basketball-boards.mesh")),
                 basketballRims(readMesh("assets/campus-basketball-rims.mesh")),
                 wallFortress(readMesh("assets/campus-wall-fortress.mesh")),
                 wallActivity(readMesh("assets/campus-wall-activity.mesh")),
                 activityFoundation(readMesh("assets/campus-activity-foundation.mesh")),
                 teachersFoundation(readMesh("assets/campus-teachers-foundation.mesh")),
                 northsportsFoundation(readMesh("assets/campus-northsports-foundation.mesh")),
                 southdormFoundation(readMesh("assets/campus-southdorm-foundation.mesh")),
                 windzhaiFoundation(readMesh("assets/campus-windzhai-foundation.mesh")),
                 huiwenroadFoundation(readMesh("assets/campus-huiwenroad-foundation.mesh")),
                 facultyroadFoundation(readMesh("assets/campus-facultyroad-foundation.mesh")),
                 wallHuiwen(readMesh("assets/campus-wall-huiwen.mesh")),
                 huiwenDetail(readMesh("assets/campus-huiwen-detail.mesh")),
                 huiwenTerrace(readMesh("assets/campus-huiwen-terrace.mesh")),
                 huiwenRed(readMesh("assets/campus-huiwen-red.mesh")),
                 huiwenSteps(readMesh("assets/campus-huiwen-steps.mesh")),
                 zhiyiShell(readMesh("assets/campus-zhiyi-shell.mesh")), zhiyiGlass(readMesh("assets/campus-zhiyi-glass.mesh")),
                 zhiyiVents(readMesh("assets/campus-zhiyi-vents.mesh")), zhiyiRails(readMesh("assets/campus-zhiyi-rails.mesh")),
                 zhiyiPodium(readMesh("assets/campus-zhiyi-podium.mesh")),
                 zhiyiSkywalk(readMesh("assets/campus-zhiyi-skywalk.mesh")), zhiyiSkyRails(readMesh("assets/campus-zhiyi-skywalk-rails.mesh")),
                 zhizhenShell(readMesh("assets/campus-zhizhen-shell.mesh")), zhizhenGlass(readMesh("assets/campus-zhizhen-glass.mesh")),
                 zhizhenSide(readMesh("assets/campus-zhizhen-side.mesh")), zhizhenFrame(readMesh("assets/campus-zhizhen-frame.mesh")),
                 zhizhenPodium(readMesh("assets/campus-zhizhen-podium.mesh")),
                 zhizhiShell(readMesh("assets/campus-zhizhi-shell.mesh")),
                 zhizhiGlass(readMesh("assets/campus-zhizhi-glass.mesh")),
                 zhizhiFrame(readMesh("assets/campus-zhizhi-frame.mesh")),
                 l4Glass(readMesh("assets/campus-l4-glass.mesh")), l4Frame(readMesh("assets/campus-l4-frame.mesh")),
                 l4Garden(readMesh("assets/campus-l4-garden.mesh")), l4Equipment(readMesh("assets/campus-l4-equipment.mesh")),
                 zhigongShell(readMesh("assets/campus-zhigong-shell.mesh")), zhigongGlass(readMesh("assets/campus-zhigong-glass.mesh")),
                 zhigongDark(readMesh("assets/campus-zhigong-dark.mesh")), zhigongPerforated(readMesh("assets/campus-zhigong-perforated.mesh")),
                 zhigongLouvers(readMesh("assets/campus-zhigong-louvers.mesh")), zhigongRails(readMesh("assets/campus-zhigong-rails.mesh")),
                 zhigongSteps(readMesh("assets/campus-zhigong-steps.mesh")), zhigongSign(readMesh("assets/campus-zhigong-sign.mesh")),
                 zhigongCarsLight(readMesh("assets/campus-zhigong-cars-light.mesh")), zhigongCarsDark(readMesh("assets/campus-zhigong-cars-dark.mesh")),
                 zhigongCarsGlass(readMesh("assets/campus-zhigong-cars-glass.mesh")), zhigongCarsWheels(readMesh("assets/campus-zhigong-cars-wheels.mesh")),
                 zhigongChargers(readMesh("assets/campus-zhigong-chargers.mesh")),
                 studioShell(readMesh("assets/campus-studio-shell.mesh")), studioGlass(readMesh("assets/campus-studio-glass.mesh")),
                 studioSign(readMesh("assets/campus-studio-sign.mesh")), studioColumns(readMesh("assets/campus-studio-columns.mesh")),
                 zhiliGlass(readMesh("assets/campus-zhili-glass.mesh")), zhiliFrame(readMesh("assets/campus-zhili-frame.mesh")),
                 zhiliSide(readMesh("assets/campus-zhili-side.mesh")),
                 zhiliAShell(readMesh("assets/campus-zhili-a-shell.mesh")),
                 zhiliAGlass(readMesh("assets/campus-zhili-a-glass.mesh")),
                 zhiliAFrame(readMesh("assets/campus-zhili-a-frame.mesh")),
                 zhiliASign(readMesh("assets/campus-zhili-a-sign.mesh")),
                 zhiliHighWalkways(readMesh("assets/campus-zhili-high-walkways.mesh")),
                 zhiliHighRails(readMesh("assets/campus-zhili-high-walkway-rails.mesh")),
                 zhiliCrown(readMesh("assets/campus-zhili-crown.mesh")),
                 zhiliCrownGlass(readMesh("assets/campus-zhili-crown-glass.mesh")),
                 zhiliCrownFrame(readMesh("assets/campus-zhili-crown-frame.mesh")),
                 huicuiShell(readMesh("assets/campus-huicui-shell.mesh")), huicuiGlass(readMesh("assets/campus-huicui-glass.mesh")),
                 huicuiUpper(readMesh("assets/campus-huicui-upper.mesh")), huicuiStage(readMesh("assets/campus-huicui-stage.mesh")),
                 huicuiFrames(readMesh("assets/campus-huicui-frames.mesh")), huicuiSkylights(readMesh("assets/campus-huicui-skylights.mesh")),
                 huiziShell(readMesh("assets/campus-huizi-shell.mesh")), huiziGround(readMesh("assets/campus-huizi-ground.mesh")),
                 huiziCanopy(readMesh("assets/campus-huizi-canopy.mesh")), huiziPaving(readMesh("assets/campus-huizi-paving.mesh")),
                 huiziRoof(readMesh("assets/campus-huizi-roof.mesh")), huiziEquipment(readMesh("assets/campus-huizi-equipment.mesh")),
                 huiyuanShell(readMesh("assets/campus-huiyuan-shell.mesh")), huiyuanGround(readMesh("assets/campus-huiyuan-ground.mesh")),
                 huiyuanSign(readMesh("assets/campus-huiyuan-sign.mesh")), huiyuanMotto(readMesh("assets/campus-huiyuan-motto.mesh")),
                 huiyuanPaving(readMesh("assets/campus-huiyuan-paving.mesh")), huiyuanPlanting(readMesh("assets/campus-huiyuan-planting.mesh")),
                 huiyuanEntryRoof(readMesh("assets/campus-huiyuan-entry-roof.mesh")),
                 yanhuiRoof(readMesh("assets/campus-yanhui-roof.mesh")), yanhuiUnder(readMesh("assets/campus-yanhui-under.mesh")),
                 yanhuiSteel(readMesh("assets/campus-yanhui-steel.mesh")), yanhuiStone(readMesh("assets/campus-yanhui-stone.mesh")),
                 yanhuiFloor(readMesh("assets/campus-yanhui-floor.mesh")), yanhuiSeats(readMesh("assets/campus-yanhui-seats.mesh")),
                 yanhuiSeatDetail(readMesh("assets/campus-yanhui-seat-detail.mesh")),
                 yanhuiStage(readMesh("assets/campus-yanhui-stage.mesh")), yanhuiScreen(readMesh("assets/campus-yanhui-screen.mesh")),
                 yanhuiSteps(readMesh("assets/campus-yanhui-steps.mesh")), yanhuiSeams(readMesh("assets/campus-yanhui-seams.mesh")),
                 yanhuiPineTrunks(readMesh("assets/campus-yanhui-pine-trunks.mesh")), yanhuiPineCrowns(readMesh("assets/campus-yanhui-pine-crowns.mesh")),
                 huixingSteps(readMesh("assets/campus-huixing-steps.mesh")),
                 huixingHedges(readMesh("assets/campus-huixing-hedges.mesh")),
                 stoneShell(readMesh("assets/campus-stone-shell.mesh")), stoneFlat(readMesh("assets/campus-stone-flat.mesh")),
                 stonePlaza(readMesh("assets/campus-stone-plaza.mesh")), southPool(readMesh("assets/campus-south-pool.mesh")),
                 stoneCanopies(readMesh("assets/campus-stone-canopies.mesh")), stoneRails(readMesh("assets/campus-stone-rails.mesh")),
                 yuanpingShell(readMesh("assets/campus-yuanping-shell.mesh")), yuanpingBase(readMesh("assets/campus-yuanping-base.mesh")),
                 yuanpingFoundation(readMesh("assets/campus-yuanping-foundation.mesh")),
                 yuanpingRoof(readMesh("assets/campus-yuanping-roof.mesh")), yuanpingTruss(readMesh("assets/campus-yuanping-truss.mesh")),
                 yuanpingRelief(readMesh("assets/campus-yuanping-relief.mesh")), yuanpingSeal(readMesh("assets/campus-yuanping-seal.mesh")),
                 yuanpingGlass(readMesh("assets/campus-yuanping-glass.mesh")), yuanpingSteps(readMesh("assets/campus-yuanping-steps.mesh")),
                 yuanpingName(readMesh("assets/campus-yuanping-name.mesh")),
                 yuanpingGarden(readMesh("assets/campus-yuanping-garden.mesh")),
                 yuanpingEntryPaving(readMesh("assets/campus-yuanping-entry-paving.mesh")),
                 beixiaShell(readMesh("assets/campus-beixia-shell.mesh")), beixiaRoof(readMesh("assets/campus-beixia-roof.mesh")),
                 beixiaFrame(readMesh("assets/campus-beixia-frame.mesh")), beixiaSeats(readMesh("assets/campus-beixia-seats.mesh")),
                 beixiaDeck(readMesh("assets/campus-beixia-deck.mesh")), beixiaLanes(readMesh("assets/campus-beixia-lanes.mesh")),
                 beixiaPractice(readMesh("assets/campus-beixia-practice.mesh")), beixiaCoping(readMesh("assets/campus-beixia-coping.mesh")),
                 beixiaUnder(readMesh("assets/campus-beixia-under.mesh")),
                 golfFloor(readMesh("assets/campus-golf-floor.mesh")), golfRoof(readMesh("assets/campus-golf-roof.mesh")),
                 golfFrame(readMesh("assets/campus-golf-frame.mesh")), golfMats(readMesh("assets/campus-golf-mats.mesh")),
                 golfSigns(readMesh("assets/campus-golf-signs.mesh")),
                 sundialFace(readMesh("assets/campus-sundial-face.mesh")), sundialStone(readMesh("assets/campus-sundial-stone.mesh")),
                 sundialPointer(readMesh("assets/campus-sundial-pointer.mesh")), timePaving(readMesh("assets/campus-time-paving.mesh")),
                 timeCurb(readMesh("assets/campus-time-curb.mesh")), timeHedges(readMesh("assets/campus-time-hedges.mesh")),
                 timeHedgeLeaves(readMesh("assets/campus-time-hedge-leaves.mesh")),
                 wallZhai(readMesh("assets/campus-wall-zhai.mesh")),
                 zhaiStone(readMesh("assets/campus-zhai-stone.mesh")),
                 zhaiPaving(readMesh("assets/campus-zhai-paving.mesh")),
                 alumniGarden(readMesh("assets/campus-alumni-garden.mesh")),
                 lideAsset(CampusAsset::loadOptional("assets/modeling/lide-gate")),
                 gates("assets/campus-gates.tsv",lideAsset ? "lide" : ""),qiaoyuanDorms("assets/campus-qiaoyuan-dorms.tsv"),
                 southDorms("assets/campus-south-dorms.tsv"), southSpine("assets/campus-south-spine.tsv"),
                 scene(0), depth(0), sky(0), shadowFbo(0), shadowTexture(0), emptyVao(0), nearInstanceVbo(0), farInstanceVbo(0),
                 nearTreeCount(0), farTreeCount(0), depthPass(false) {
        if (lideAsset) lideAsset->upload();
        scene = loadShaderProgram("shaders/scene.vert.glsl", "shaders/scene.frag.glsl");
        depth = loadShaderProgram("shaders/depth.vert.glsl", "shaders/depth.frag.glsl");
        sky = loadShaderProgram("shaders/sky.vert.glsl", "shaders/sky.frag.glsl");
        for (int i = 0; i < TREE_BARK; ++i) textures[i] = loadTexture(std::string("assets/") + textureFiles[i] + ".ppm");
        textures[TREE_BARK] = loadRgbaTexture("assets/tree-bark.rgba");
        textures[TREE_LEAVES] = loadRgbaTexture("assets/tree-leaves.rgba");
        textures[PALM_ATLAS] = loadRgbaTexture("assets/palm.rgba",true);
        textures[TREE_CANOPY] = loadRgbaTexture("assets/campus-canopy.rgba");
        textures[TREE_CLUSTER] = loadRgbaTexture("assets/campus-leaf-cluster.rgba");
        textures[CAMPUS_GROUND] = loadTexture("assets/campus-ground.ppm");
        textures[FACADE_CREAM] = loadTexture("assets/facade-cream.ppm");
        textures[FACADE_RED] = loadTexture("assets/facade-red.ppm");
        textures[FACADE_HUIYAN] = loadTexture("assets/facade-huiyan.ppm");
        textures[STADIUM_SURFACE] = loadTexture("assets/stadium-surface.ppm");
        textures[STADIUM_ENTRY_PAVING] = loadTexture("assets/stadium-entry-paving.ppm");
        textures[NORTH_LIBRARY_ENTRY_PAVING] = loadTexture("assets/north-library-entry-paving.ppm");
        textures[STADIUM_SEAT] = loadTexture("assets/stadium-seat.ppm");
        textures[FACADE_CITY] = loadTexture("assets/facade-city.ppm");
        textures[FACADE_STRIP] = loadTexture("assets/facade-strip.ppm");
        textures[FACADE_PERFORATED] = loadTexture("assets/facade-perforated.ppm");
        textures[FACADE_TENG] = loadTexture("assets/facade-teng.ppm");
        textures[FACADE_ATRIUM] = loadTexture("assets/facade-atrium.ppm");
        textures[HUIYUAN_SIGN] = loadTexture("assets/huiyuan-sign.ppm");
        textures[HUIYUAN_MOTTO] = loadTexture("assets/huiyuan-motto.ppm");
        textures[ARCHIVE_SEAL] = loadTexture("assets/archive-seal.ppm");
        textures[BASKETBALL_COURT] = loadTexture("assets/basketball-court.ppm");
        textures[TENNIS_COURT] = loadTexture("assets/tennis-court.ppm");
        textures[SUNDIAL_FACE] = loadTexture("assets/sundial-face.ppm");
        textures[SUNDIAL_STONE] = loadTexture("assets/sundial-stone.ppm");
        textures[FACADE_QIAOMU] = loadTexture("assets/facade-qiaomu.ppm");
        textures[FACADE_QIAOSEN] = loadTexture("assets/facade-qiaosen.ppm");
        textures[FACADE_ZHAI] = loadTexture("assets/facade-zhai.ppm");
        textures[FACADE_GATE] = loadTexture("assets/facade-gate.ppm");
        textures[GATE_SIGN] = loadTexture("assets/gate-sign.ppm");
        textures[GATE_ADDRESS] = loadTexture("assets/gate-address.ppm");
        textures[ZHIYUAN_WORDS] = loadRgbaTexture("assets/zhiyuan-sign-words.rgba");
        textures[ZHIYUAN_BASE] = loadTexture("assets/zhiyuan-sign-base.ppm");
        textures[SPORTS_FENCE] = loadRgbaTexture("assets/sports-fence.rgba");
        textures[FACADE_FORTRESS] = loadTexture("assets/facade-stone-fortress.ppm");
        textures[STONE_PLAZA] = loadTexture("assets/stone-plaza.ppm");
        textures[SOUTH_POOL] = loadTexture("assets/south-pool.ppm");
        textures[YUANPING_RELIEF] = loadTexture("assets/yuanping-relief.ppm");
        textures[YUANPING_SEAL] = loadTexture("assets/yuanping-seal.ppm");
        textures[YUANPING_NAME] = loadTexture("assets/yuanping-name.ppm");
        textures[YUANPING_ENTRY_PAVING] = loadTexture("assets/yuanping-entry-paving.ppm");
        textures[GOLF_DISTANCE] = loadTexture("assets/golf-distance.ppm");
        textures[NORTH_CITY_GROUND] = loadTexture("assets/campus-north-city-ground.ppm");
        textures[HUIZI_ROOF] = loadTexture("assets/huizi-roof.ppm");
        textures[HUIZI_EQUIPMENT] = loadTexture("assets/huizi-equipment.ppm");
        textures[FACADE_ACTIVITY] = loadTexture("assets/facade-activity-red.ppm");
        textures[FACADE_HUIWEN] = loadTexture("assets/facade-huiwen.ppm");
        textures[FACADE_HUIWEN_RED] = loadTexture("assets/facade-huiwen-red.ppm");
        textures[FACADE_HUIXING] = loadTexture("assets/facade-huixing.ppm");
        textures[FACADE_HUIXING_CORE] = loadTexture("assets/facade-huixing-core.ppm");
        textures[FACADE_HUIXING_PODIUM] = loadTexture("assets/facade-huixing-podium.ppm");
        textures[HUIXING_ROOF] = loadTexture("assets/huixing-podium-roof.ppm");
        textures[FACADE_HUIZHI] = loadTexture("assets/facade-huizhi.ppm");
        textures[HUIZHI_ENTRANCE] = loadTexture("assets/huizhi-entrance.ppm");
        textures[FACADE_ZHIYI_VENT] = loadTexture("assets/facade-zhiyi-vent.ppm");
        textures[FACADE_ZHIYI_PODIUM] = loadTexture("assets/facade-zhiyi-podium.ppm");
        textures[STUDIO_SIGN] = loadTexture("assets/studio-sign.ppm");
        textures[FACADE_ZHILI_SIDE] = loadTexture("assets/facade-zhili-side.ppm");
        textures[ZHILI_A_SIGN] = loadTexture("assets/zhili-a-sign.ppm");
        textures[ZHIGONG_LOUVER] = loadTexture("assets/zhigong-louver.ppm");
        textures[ZHIGONG_SIGN] = loadTexture("assets/zhigong-sign.ppm");
        textures[CANTEEN_SIGN] = loadTexture("assets/canteen-sign.ppm");
        textures[DUJUAN_BOARDWALK] = loadTexture("assets/dujuan-boardwalk.ppm");
        textures[CAMPUS_ASPHALT] = loadTexture("assets/campus-asphalt.ppm");
        textures[QIAOYUAN_GARDEN_WALK] = loadTexture("assets/qiaoyuan-garden-walk.ppm");
        textures[QIAOYUAN_BRICK] = loadTexture("assets/qiaoyuan-brick.ppm");
        textures[QIAOYUAN_GARDEN_WORDS] = loadRgbaTexture("assets/qiaoyuan-garden-words.rgba");
        textures[XIYUAN_SIGNS] = loadTexture("assets/xiyuan-shop-signs.ppm");
        textures[XIYUAN_FACADE] = loadTexture("assets/facade-xiyuan.ppm");
        textures[XIYUAN_PAVING] = loadTexture("assets/xiyuan-paving.ppm");
        textures[XIYUAN_WINDOW] = loadTexture("assets/xiyuan-store-window.ppm");
        textures[MANNER_WALL] = loadTexture("assets/xiyuan-manner-wall.ppm");
        textures[CAMPUS_SIDEWALK] = loadTexture("assets/campus-sidewalk.ppm");
        textures[ZHOUNAN_ZHAI_TILE] = loadTexture("assets/zhounan-zhai-tile.ppm");
        textures[SHUTTLE_SIGN] = loadTexture("assets/shuttle-sign.ppm");
        textures[WENSHAN_ANCIENT_STONE] = loadTexture("assets/wenshan-ancient-stone.ppm");
        textures[SWAN_FEATHERS] = loadTexture("assets/wenshan-swan-feathers.ppm");
        textures[KAIFENG_TILE] = loadTexture("assets/kaifeng-tiles.ppm");
        // 所有树共享两份网格，按实例变换一次提交，降低全校园的绘制开销。
        std::ifstream trees("assets/campus-trees.bin",std::ios::binary);
        std::uint32_t count = 0; trees.read(reinterpret_cast<char*>(&count),4);
        if (!trees || !count || count > 10000) throw std::runtime_error("Invalid campus tree instances");
        treePositions.resize(count);
        static_assert(sizeof(glm::vec4)==4*sizeof(float), "Unexpected instance layout");
        trees.read(reinterpret_cast<char*>(&treePositions[0]),static_cast<std::streamsize>(treePositions.size()*sizeof(glm::vec4)));
        if (!trees) throw std::runtime_error("Truncated campus tree instances");
        // 可靠标注的树点复用既有实例网格；零个点是有效的，不能把示意点当作实测。
        std::ifstream annotated("assets/campus-annotated-trees.bin",std::ios::binary);
        std::uint32_t extra=0; annotated.read(reinterpret_cast<char*>(&extra),4);
        if (!annotated || extra>10000 || count+extra>10000) throw std::runtime_error("Invalid annotated tree instances");
        if (extra) {
            treePositions.resize(count+extra);
            annotated.read(reinterpret_cast<char*>(&treePositions[count]),static_cast<std::streamsize>(extra*sizeof(glm::vec4)));
            if (!annotated) throw std::runtime_error("Truncated annotated tree instances");
        }
        glGenBuffers(1,&nearInstanceVbo); glGenBuffers(1,&farInstanceVbo);
        Mesh* treeParts[] = {&treeBark,&treeLeaves,&nearLeafCards,&farTrunk,&farCrown,&farLeafCards};
        for (size_t i=0;i<6;++i) {
            glBindVertexArray(treeParts[i]->vao);
            glBindBuffer(GL_ARRAY_BUFFER,i<3 ? nearInstanceVbo : farInstanceVbo);
            glEnableVertexAttribArray(3); glVertexAttribPointer(3,4,GL_FLOAT,GL_FALSE,sizeof(glm::vec4),NULL);
            glVertexAttribDivisor(3,1);
        }
        glBindVertexArray(0);
        glGenVertexArrays(1, &emptyVao);
        glGenFramebuffers(1, &shadowFbo);
        glGenTextures(1, &shadowTexture); glBindTexture(GL_TEXTURE_2D, shadowTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, SHADOW_SIZE, SHADOW_SIZE, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        const float border[] = {1,1,1,1}; glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border);
        glBindFramebuffer(GL_FRAMEBUFFER, shadowFbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, shadowTexture, 0);
        glDrawBuffer(GL_NONE); glReadBuffer(GL_NONE);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            throw std::runtime_error("Shadow framebuffer is incomplete");
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
    ~Renderer() {
        glDeleteTextures(TEXTURE_COUNT, textures); glDeleteTextures(1, &shadowTexture);
        glDeleteFramebuffers(1, &shadowFbo); glDeleteVertexArrays(1, &emptyVao);
        glDeleteBuffers(1,&nearInstanceVbo); glDeleteBuffers(1,&farInstanceVbo);
        glDeleteProgram(scene); glDeleteProgram(depth); glDeleteProgram(sky);
    }

    void draw(const Mesh& mesh, const glm::mat4& model, Texture tex, const glm::vec3& color,
              const glm::vec2& repeat = glm::vec2(1), int surface = 0) {
        GLuint program = depthPass ? depth : scene;
        glUniform1i(glGetUniformLocation(program, "useInstancing"),0);
        glUniformMatrix4fv(glGetUniformLocation(program, "model"), 1, GL_FALSE, glm::value_ptr(model));
        glUniform1i(glGetUniformLocation(program, "alphaCutout"),
                    tex == SPORTS_FENCE || tex == ZHIYUAN_WORDS || tex == QIAOYUAN_GARDEN_WORDS ? 2 : tex == TREE_LEAVES || tex == TREE_CANOPY || tex == TREE_CLUSTER ? 1 : 0);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, textures[tex]);
        if (!depthPass) {
            glUniform3fv(glGetUniformLocation(scene, "baseColor"), 1, glm::value_ptr(color));
            glUniform2fv(glGetUniformLocation(scene, "textureScale"), 1, glm::value_ptr(repeat));
            glUniform1i(glGetUniformLocation(scene, "surfaceType"), surface);
        }
        mesh.draw();
    }

    void prepareTrees(const glm::vec3& eye) {
        std::vector<glm::vec4> nearTrees, farTrees;
        nearTrees.reserve(treePositions.size()); farTrees.reserve(treePositions.size());
        for (size_t i=0;i<treePositions.size();++i) {
            glm::vec3 p(treePositions[i]);
            if (glm::length(p-eye)<140.0f) nearTrees.push_back(treePositions[i]);
            else farTrees.push_back(treePositions[i]);
        }
        nearTreeCount=static_cast<GLsizei>(nearTrees.size());
        farTreeCount=static_cast<GLsizei>(farTrees.size());
        glBindBuffer(GL_ARRAY_BUFFER,nearInstanceVbo);
        glBufferData(GL_ARRAY_BUFFER,nearTrees.size()*sizeof(glm::vec4),nearTrees.empty() ? NULL : &nearTrees[0],GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER,farInstanceVbo);
        glBufferData(GL_ARRAY_BUFFER,farTrees.size()*sizeof(glm::vec4),farTrees.empty() ? NULL : &farTrees[0],GL_DYNAMIC_DRAW);
    }

    void trees() {
        GLuint program = depthPass ? depth : scene;
        glUniform1i(glGetUniformLocation(program,"useInstancing"),1);
        glm::mat4 identity(1.0f);
        glUniformMatrix4fv(glGetUniformLocation(program,"model"),1,GL_FALSE,glm::value_ptr(identity));
        const Mesh* parts[] = {&treeBark,&treeLeaves,&nearLeafCards,&farTrunk,&farCrown,&farLeafCards};
        const Texture materials[] = {TREE_BARK,TREE_CANOPY,TREE_CLUSTER,TREE_BARK,TREE_CANOPY,TREE_CLUSTER};
        for (int i=0;i<6;++i) {
            GLsizei count=i<3 ? nearTreeCount : farTreeCount;
            if (!count) continue;
            glUniform1i(glGetUniformLocation(program,"alphaCutout"),materials[i]==TREE_CLUSTER ? 1 : 0);
            glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D,textures[materials[i]]);
            if (!depthPass) {
                if (materials[i]==TREE_BARK) glUniform3f(glGetUniformLocation(scene,"baseColor"),.68f,.83f,.88f);
                else glUniform3f(glGetUniformLocation(scene,"baseColor"),.83f,.91f,.80f);
                glUniform2f(glGetUniformLocation(scene,"textureScale"),1,1);
                glUniform1i(glGetUniformLocation(scene,"surfaceType"),materials[i]==TREE_BARK ? 0 : 2);
            }
            glBindVertexArray(parts[i]->vao);
            glDrawArraysInstanced(GL_TRIANGLES,0,parts[i]->count,count);
        }
        glUniform1i(glGetUniformLocation(program,"useInstancing"),0);
    }
};

glm::mat4 moved(const glm::mat4& m, float x, float y, float z) { return glm::translate(m, glm::vec3(x,y,z)); }
glm::mat4 scaled(const glm::mat4& m, float x, float y, float z) { return glm::scale(m, glm::vec3(x,y,z)); }
glm::mat4 turned(const glm::mat4& m, float degrees, float x, float y, float z) { return glm::rotate(m, glm::radians(degrees), glm::vec3(x,y,z)); }

void box(Renderer& r, const glm::mat4& parent, glm::vec3 p, glm::vec3 size, Texture tex,
         glm::vec3 color = glm::vec3(1), glm::vec2 repeat = glm::vec2(1)) {
    r.draw(r.cube, scaled(glm::translate(parent,p),size.x,size.y,size.z), tex, color, repeat);
}
void cylinder(Renderer& r, const glm::mat4& parent, glm::vec3 p, glm::vec3 size, Texture tex,
              glm::vec3 color = glm::vec3(1), int surface = 0) {
    r.draw(r.cylinder, scaled(glm::translate(parent,p),size.x,size.y,size.z), tex, color, glm::vec2(1), surface);
}

// 层次实例：校园 → 汇典楼 → 楼层 → 窗户 → 窗框，子构件继承全部父变换。
void northLibrary(Renderer& r, const glm::mat4& campus) {
    glm::mat4 building = campus;
    const glm::vec3 ivory(.94f,.96f,.94f), pale(.84f,.88f,.87f);
    // OSM 轮廓与早期图纸均接近 60 米方形；北馆总高参考校方 25 米数据。
    box(r,building,{0,-5,0},{60,10,60},STONE,{.83f,.85f,.82f});
    box(r,building,{0,.12f,0},{60.4f,.24f,60.4f},STONE);
    // 上部书库由立柱抬起，首层幕墙退到柱列后方，形成照片中通透的入口灰空间。
    // 门前幕墙拆为面板，中央净开口与独立门框对齐，避免把门框叠在实玻璃盒上。
    const float doorHalf=2.2f;
    for (int side=-1;side<=1;side+=2) {
        box(r,building,{side*(24+doorHalf)*.5f,1.875f,22.42f},{24-doorHalf,3.75f,.16f},GLASS,{.48f,.57f,.58f},{5.5f,1});
        box(r,building,{side*24.0f,1.875f,8},{.16f,3.75f,29},GLASS,{.48f,.57f,.58f},{7,1});
    }
    box(r,building,{0,3.345f,22.42f},{4.4f,.81f,.16f},GLASS,{.48f,.57f,.58f},{1,1});
    box(r,building,{0,1.875f,-6.5f},{48,3.75f,.16f},GLASS,{.48f,.57f,.58f},{12,1});
    for (int i=-4;i<=4;++i)
        box(r,building,{i*6.5f,2.17f,27.5f},{.65f,4.34f,.65f},CONCRETE,pale);
    for (int side=-1;side<=1;side+=2)
        for (int i=-2;i<=2;++i)
            box(r,building,{side*28.5f,2.17f,i*10.0f},{.65f,4.34f,.65f},CONCRETE,pale);
    // 二层平台与门前外挑平台：两侧真实连廊在生成网格中与此标高接合。
    // 二层板与U形侧梯的开口已由同一局部生成器制作，稍后以world网格绘制。
    box(r,building,{0,4.72f,39.3f},{35,.75f,.38f},CONCRETE,ivory);
    box(r,building,{0,5.12f,39.25f},{34,.10f,.55f},GRASS,{.78f,.94f,.72f});
    box(r,building,{0,4.63f,39.52f},{12,.65f,.10f},HUIDIAN);

    // 入口柱延续至三层底，侧梯和路线均避开此柱列。
    for (int i=-4;i<=4;++i)
        box(r,building,{i*6.5f,6.15f,27.5f},{.65f,3.6f,.65f},CONCRETE,pale);
    // 二层两侧保留门洞与地图连廊相接；三层以上为封闭的方形书库外壳。
    box(r,building,{0,15.8f,0},{59.2f,15.7f,59.2f},GLASS,{.73f,.81f,.84f},{15,4});
    for (int side=-1;side<=1;side+=2) {
        if (side<0) box(r,building,{0,6.15f,-29.6f},{59.2f,3.6f,.16f},GLASS,{.73f,.81f,.84f},{15,1});
        else {
            for (int wing=-1;wing<=1;wing+=2)
                box(r,building,{wing*(29.6f+doorHalf)*.5f,6.15f,22.42f},{29.6f-doorHalf,3.6f,.16f},GLASS,{.73f,.81f,.84f},{7,1});
            box(r,building,{0,7.55f,22.42f},{4.4f,.80f,.16f},GLASS,{.73f,.81f,.84f},{1,1});
        }
        box(r,building,{side*29.6f,6.15f,-7.8f},{.16f,3.6f,43.6f},GLASS,{.73f,.81f,.84f},{11,1});
        box(r,building,{side*29.6f,6.15f,21.25f},{.16f,3.6f,2.5f},GLASS,{.73f,.81f,.84f},{1,1});
    }
    box(r,building,{0,23.8f,0},{60,.30f,60},CONCRETE,ivory);
    // 航拍和用户模型均可见分段屋顶围护与中央深色采光面；尺寸为视觉估计。
    box(r,building,{0,23.99f,-3},{17,.08f,10},GLASS,{.46f,.55f,.58f},{5,3});
    for (int side=-1;side<=1;side+=2) {
        box(r,building,{side*9.2f,24.35f,-3},{.45f,.95f,11.4f},CONCRETE,pale);
        box(r,building,{0,24.35f,-3+side*5.5f},{18.8f,.95f,.45f},CONCRETE,pale);
    }
    box(r,building,{0,24.35f,-18.0f},{22,1.1f,2.2f},CONCRETE,pale);
    box(r,building,{0,24.35f,10.8f},{22,1.1f,2.2f},CONCRETE,pale);
    // 四面均为凸出的白色窗间墙，窄窗随楼层分格。
    for (int i=0;i<16;++i) {
        float p=-28.2f+i*(56.4f/15.0f);
        for (int side=-1;side<=1;side+=2) {
            box(r,building,{p,side>0 ? 15.8f : 14.0f,side*29.65f},
                {2.30f,side>0 ? 15.7f : 19.3f,.40f},CONCRETE,ivory,{1,5});
            box(r,building,{side*29.65f,15.8f,p},{.40f,15.7f,2.30f},CONCRETE,ivory,{1,4});
            if (p < 13.0f || p > 20.8f)
                box(r,building,{side*29.65f,6.15f,p},{.40f,3.6f,2.30f},CONCRETE,ivory);
        }
    }
    // 窗框局部网格继承校园与北馆的层次变换。
    r.draw(r.northLibraryWindows,building,WHITE,{.72f,.75f,.73f});
}

// 南馆以玻璃幕墙、白色竖框和左右错落的体块为主要特征。
void southLibrary(Renderer& r, const glm::mat4& campus) {
    glm::mat4 building = turned(campus, 180, 0,1,0);
    glm::vec3 ivory(1.18f,1.18f,1.13f), rail(.93f,.96f,.91f);
    box(r,building,{0,-4,0},{36,8,15},STONE,{.83f,.85f,.82f});
    box(r, building, {0,.35f,0}, {37,.7f,16}, STONE);
    for (int side = 0; side < 2; ++side) {
        float x = side == 0 ? -6.4f : 13.5f;
        float width = side == 0 ? 22.0f : 8.0f;
        int columns = side == 0 ? 11 : 4;
        glm::mat4 block = moved(building,x,0,0);
        box(r, block, {0,7.3f,0}, {width,13.6f,12}, CONCRETE, ivory);
        box(r, block, {0,7.4f,6.08f}, {width-.6f,12.2f,.18f}, FACADE_HUIZHI, glm::vec3(1.50f), {float(columns),1});
        box(r, block, {0,7.4f,-6.08f}, {width-.6f,12.2f,.18f}, FACADE_HUIZHI, glm::vec3(1.50f), {float(columns),1});
        box(r, block, {0,14.25f,0}, {width+.2f,.4f,12.4f}, CONCRETE, ivory);
        for (int i = 0; i <= columns; ++i) {
            float offset = -width*.5f + .35f + i * ((width-.7f)/columns);
            box(r, block, {offset,7.4f,6.2f}, {.16f,12.4f,.24f}, CONCRETE, ivory);
            box(r, block, {offset,7.4f,-6.2f}, {.16f,12.4f,.24f}, CONCRETE, ivory);
        }
        for (int floor = 0; floor <= 5; ++floor) {
            glm::mat4 level = moved(block,0,1.3f + floor*2.45f,6.22f);
            box(r, level, {0,0,0}, {width-.5f,.16f,.25f}, CONCRETE, rail);
            box(r, block, {0,1.3f + floor*2.45f,-6.22f}, {width-.5f,.16f,.25f}, CONCRETE, rail);
        }
        for (int sideFace=-1; sideFace<=1; sideFace+=2) {
            box(r, block, {sideFace*(width*.5f+.06f),7.4f,0}, {.18f,12.2f,11.3f}, FACADE_HUIZHI, glm::vec3(1.50f), {5,1});
            for (int floor=0; floor<=5; ++floor)
                box(r, block, {sideFace*(width*.5f+.16f),1.3f+floor*2.45f,0}, {.25f,.16f,11.4f}, CONCRETE, rail);
            for (int j=-2; j<=2; ++j)
                box(r, block, {sideFace*(width*.5f+.16f),7.4f,j*2.1f}, {.25f,12.2f,.16f}, CONCRETE, ivory);
        }
    }
    // 中部为低层入口，上方两层跨接走廊与透明栏杆保留开放庭院。
    box(r, building, {7.05f,2.2f,0}, {5.3f,3.7f,9}, GLASS, {.73f,.84f,.83f},{3,1});
    box(r, building, {7.05f,4.2f,0}, {5.3f,.4f,11.8f}, CONCRETE, ivory);
    for (float level : {8.0f,13.55f}) {
        box(r,building,{7.05f,level,0},{5.3f,.28f,7.2f},CONCRETE,ivory);
        for (int face=-1;face<=1;face+=2) {
            box(r,building,{7.05f,level+1.0f,face*3.48f},{5.3f,.07f,.07f},WHITE,rail);
            for (int post=0;post<=4;++post)
                box(r,building,{4.45f+post*1.3f,level+.5f,face*3.48f},{.06f,1.0f,.07f},WHITE,rail);
        }
    }
    // 航拍可见的屋顶机房和入口前出挑雨棚。
    box(r,building,{-7,14.7f,-2.5f},{7.5f,.9f,4.5f},CONCRETE,ivory);
    box(r,building,{13.5f,14.7f,-1.7f},{5,.9f,3.5f},CONCRETE,ivory);
    box(r,building,{-3.5f,3.8f,7.2f},{15,.26f,2.0f},CONCRETE,ivory);
    box(r, building, {7.05f,2.55f,4.58f}, {4.8f,1.15f,.10f}, HUIZHI_ENTRANCE,glm::vec3(1.30f));
    // 入口照片中台阶上有矩形花池，尺寸和种植方式按外观近似。
    box(r,building,{7.05f,.70f,7.8f},{4.8f,.65f,1.25f},STONE,{.90f,.90f,.86f});
    box(r,building,{7.05f,1.06f,7.8f},{4.55f,.25f,1.02f},WHITE,{.35f,.49f,.24f});
    for (int i = 0; i < 5; ++i)
        box(r, building, {7.05f,.12f+i*.12f,8.5f-i*.35f}, {11,.22f,2.0f}, STONE);
}

// 一段汇星楼外墙；不同立面分段保留真实的空中花园洞口。
void huixingWall(Renderer& r, const glm::mat4& parent, bool alongX, float fixed,
                 float start, float end, float bottom, float top) {
    float length=end-start, height=top-bottom, middle=(start+end)*.5f;
    glm::vec3 position=alongX ? glm::vec3(middle,(bottom+top)*.5f,fixed)
                             : glm::vec3(fixed,(bottom+top)*.5f,middle);
    glm::vec3 size=alongX ? glm::vec3(length,height,2.4f) : glm::vec3(2.4f,height,length);
    box(r,parent,position,size,FACADE_HUIXING,glm::vec3(1),{length/6.0f,height/4.3f});
}

// 汇星楼：15 层主体、错层空中花园和玻璃交通筒；95 米来自建筑设计院资料。
void huixing(Renderer& r, const glm::mat4& campus) {
    const glm::vec3 light(.91f,.92f,.88f), trim(.69f,.72f,.68f);
    const float base=8.0f, roofY=72.5f, floorH=4.3f;
    // SKP 与实景均显示主体接近方形：东西两面为较窄的低洞口，
    // 南北两面为较宽的高洞口；具体标高和朝向仍按参考资料近似。
    const float half=26.4f, wall=2.4f, outer=half-wall*.5f;
    const float lowBottom=base+4*floorH, lowTop=base+10*floorH;
    const float highBottom=base+9*floorH, highTop=base+12*floorH;

    // 台地形裙房与屋顶采光带；其高差、格栅密度根据照片估计。
    box(r,campus,{0,1.4f,0},{106,5.2f,62},FACADE_HUIXING_PODIUM,glm::vec3(1),{18,1});
    box(r,campus,{0,4.07f,0},{106,.14f,62},HUIXING_ROOF,glm::vec3(1),{18,4});
    box(r,campus,{0,4.12f,30.8f},{106,.30f,.35f},CONCRETE,light);
    box(r,campus,{0,4.12f,-30.8f},{106,.30f,.35f},CONCRETE,light);
    box(r,campus,{52.8f,4.12f,0},{.35f,.30f,62},CONCRETE,light);
    box(r,campus,{-52.8f,4.12f,0},{.35f,.30f,62},CONCRETE,light);
    box(r,campus,{-2.5f,1.2f,-31.08f},{12,3.0f,.12f},GLASS,{.56f,.68f,.67f},{4,1});

    // 塔楼退后的首层、柱子和楼板，形成近景可见的架空入口。
    box(r,campus,{0,6.0f,0},{39,3.7f,37},FACADE_HUIXING_CORE,{.74f,.83f,.80f},{12,1});
    for (int x=-2;x<=2;++x) for (int z=-1;z<=1;z+=2)
        box(r,campus,{x*10.0f,6.0f,z*24.1f},{1.6f,3.8f,1.6f},CONCRETE,light);
    box(r,campus,{0,7.95f,0},{52.8f,.45f,52.8f},CONCRETE,light);

    // 不同立面保留错层洞口，近似宽度和高度参照用户模型及设计院照片。
    for (int sign=-1;sign<=1;sign+=2) {
        float z=sign*outer;
        huixingWall(r,campus,true,z,-half,half,base,highBottom);
        huixingWall(r,campus,true,z,-half,half,highTop,roofY);
        huixingWall(r,campus,true,z,-half,-16.8f,highBottom,highTop);
        huixingWall(r,campus,true,z,16.8f,half,highBottom,highTop);
        float x=sign*outer;
        huixingWall(r,campus,false,x,-half,half,base,lowBottom);
        huixingWall(r,campus,false,x,-half,half,lowTop,roofY);
        huixingWall(r,campus,false,x,-half,-12.0f,lowBottom,lowTop);
        huixingWall(r,campus,false,x,12.0f,half,lowBottom,lowTop);

        // 洞口不只是外墙上的一张空框：设置通向中庭的侧壁、天花及平台。
        for (int side=-1;side<=1;side+=2) {
            box(r,campus,{sign*21.45f,(lowBottom+lowTop)*.5f,side*12.0f},
                {9.9f,lowTop-lowBottom,.24f},CONCRETE,light);
            box(r,campus,{side*16.8f,(highBottom+highTop)*.5f,sign*19.8f},
                {.24f,highTop-highBottom,13.2f},CONCRETE,light);
        }
        box(r,campus,{sign*15.9f,lowBottom,0},{21.0f,.38f,24},CONCRETE,light);
        box(r,campus,{sign*21.45f,lowTop,0},{9.9f,.28f,24},CONCRETE,light);
        box(r,campus,{0,highBottom,sign*19.8f},{33.6f,.38f,13.2f},CONCRETE,light);
        box(r,campus,{0,highTop,sign*19.8f},{33.6f,.28f,13.2f},CONCRETE,light);
        box(r,campus,{sign*26.42f,lowBottom+1,0},{.10f,.10f,24},WHITE,trim);
        box(r,campus,{0,highBottom+1,sign*26.42f},{33.6f,.10f,.10f},WHITE,trim);
        for (int i=-3;i<=3;++i) {
            box(r,campus,{sign*26.42f,lowBottom+.5f,i*4.0f},{.10f,1.0f,.10f},WHITE,trim);
            box(r,campus,{i*5.6f,highBottom+.5f,sign*26.42f},{.10f,1.0f,.10f},WHITE,trim);
        }
        // “日”字形的内院立面可从低洞口望见，窗带与外立面连续。
        for (int side=-1;side<=1;side+=2)
            huixingWall(r,campus,true,side*13.2f,std::min(sign*6.2f,sign*16.5f),
                        std::max(sign*6.2f,sign*16.5f),base,highBottom);
        huixingWall(r,campus,false,sign*16.5f,-13.2f,13.2f,base,lowBottom);
        huixingWall(r,campus,false,sign*16.5f,-13.2f,13.2f,lowTop,roofY);
    }
    // 中央玻璃交通筒位于地图记录的独立分部；立面与屋顶框架分别处理。
    box(r,campus,{-.51f,(base+roofY)*.5f,1.53f},{12.44f,roofY-base,24.17f},
        FACADE_HUIXING_CORE,{.90f,.95f,.92f},{8,16});
    for (int sign=-1;sign<=1;sign+=2) {
        box(r,campus,{sign*21.6f,roofY,0},{9.6f,.42f,52.8f},CONCRETE,light);
        box(r,campus,{0,roofY,sign*19.8f},{33.6f,.42f,13.2f},CONCRETE,light);
        for (int other=-1;other<=1;other+=2)
            box(r,campus,{sign*21.0f,roofY+.8f,other*16.0f},{4,1.6f,4},CONCRETE,light);
    }
    // 塔内设备实体低于外层钢架；顶部留空以表现照片中可透光的框架。
    box(r,campus,{-.51f,80.0f,1.53f},{7.8f,15.0f,11.8f},CONCRETE,{.70f,.74f,.70f});
    r.draw(r.huixingFrame,campus,WHITE,{.31f,.37f,.36f});

}

void tree(Renderer& r, const glm::mat4& scene, float x, float z, float scale, int variant) {
    // 树干取自 Quaternius，树冠使用程序生成网格和 CC0 叶片材质。
    // 单棵树不读取实例缓冲；俯瞰时近处实例可能为空，必须暂停对应顶点属性。
    const Mesh* parts[]={&r.treeBark,&r.treeLeaves,&r.nearLeafCards};
    for (int i=0;i<3;++i) { glBindVertexArray(parts[i]->vao); glDisableVertexAttribArray(3); }
    glm::mat4 root = turned(moved(scene,x,.13f,z),variant*43.0f,0,1,0);
    root = scaled(root,scale*.68f,scale*.68f,scale*.68f);
    r.draw(r.treeBark,root,TREE_BARK,glm::vec3(.68f,.83f,.88f));
    float shade = .87f + .035f * (variant % 4);
    r.draw(r.treeLeaves,root,TREE_CANOPY,glm::vec3(shade*.83f,.91f,shade*.80f),glm::vec2(1),2);
    r.draw(r.nearLeafCards,root,TREE_CLUSTER,glm::vec3(shade*.83f,.91f,shade*.80f),glm::vec2(1),2);
    for (int i=0;i<3;++i) { glBindVertexArray(parts[i]->vao); glEnableVertexAttribArray(3); }
    glBindVertexArray(0);
}

void palm(Renderer& r, const glm::mat4& scene, float x, float z, float height, int seed) {
    // 棕榈网格来自 Quaternius 的 CC0 Pirate Kit，按场景尺度缩放。
    glm::mat4 root = turned(moved(scene,x,0,z),seed*27.0f,0,1,0);
    float scale = height / 2.69f;
    r.draw(r.palmMesh,scaled(root,scale*.64f,scale,scale*.64f),PALM_ATLAS,glm::vec3(1));
}

void royalPalm(Renderer& r, const glm::mat4& scene, float x, float z, float height, int seed) {
    // 南馆入口两侧为细长的棕榈行道树，叶片采用轻量程序网格。
    glm::mat4 root=turned(moved(scene,x,r.heights.at(x,z),z),seed*31.0f,0,1,0);
    cylinder(r,root,{0,height*.5f,0},{.30f,height,.30f},CONCRETE,{.82f,.83f,.77f});
    r.draw(r.royalPalmLeaves,moved(root,0,height,0),WHITE,{.27f,.45f,.20f},glm::vec2(1),2);
}

// 浑天仪中的环依次继承雕塑整体、用户旋转和各自倾角。
void armillary(Renderer& r, const glm::mat4& campus, const State& s) {
    cylinder(r,campus,{0,.25f,0},{6.8f,.38f,6.8f},STONE,{.84f,.86f,.83f});
    cylinder(r,campus,{0,.47f,0},{6.2f,.09f,6.2f},WATER,{.86f,.94f,1.0f},1);
    cylinder(r,campus,{0,.72f,0},{2.35f,.55f,2.35f},STONE,{.60f,.62f,.58f});
    cylinder(r,campus,{0,1.16f,0},{1.55f,.48f,1.55f},STONE,{.52f,.55f,.54f});
    glm::mat4 assembly = turned(moved(campus,0,2.55f,0),s.armillaryAngle,0,1,0);
    const glm::vec3 metal(.62f,.57f,.40f);
    r.draw(r.torus,scaled(assembly,3.1f,3.1f,3.1f),WHITE,metal);
    r.draw(r.torus,scaled(turned(assembly,90,1,0,0),3.1f,3.1f,3.1f),WHITE,metal);
    r.draw(r.torus,scaled(turned(assembly,90,0,0,1),3.1f,3.1f,3.1f),WHITE,metal);
    r.draw(r.torus,scaled(turned(assembly,32,1,0,1),2.25f,2.25f,2.25f),WHITE,{.70f,.64f,.47f});
    r.draw(r.sphere,scaled(assembly,.34f,.34f,.34f),WHITE,{.72f,.65f,.46f});
    // 八条喷泉水流由短圆柱沿抛物线连接，形成与广场水景相呼应的轮廓。
    for (int stream=0; stream<8; ++stream) {
        float angle = stream * 2.0f * PI / 8.0f;
        glm::vec3 direction(cos(angle),0,sin(angle));
        for (int step=0; step<12; ++step) {
            float a = step/12.0f, b = (step+1)/12.0f;
            glm::vec3 p = direction*(2.65f-1.35f*a) + glm::vec3(0,.58f+2.1f*4*a*(1-a),0);
            glm::vec3 q = direction*(2.65f-1.35f*b) + glm::vec3(0,.58f+2.1f*4*b*(1-b),0);
            glm::vec3 delta = q-p, up(0,1,0), axis = glm::cross(up,glm::normalize(delta));
            float rotation = glm::degrees(acos(glm::clamp(glm::dot(up,glm::normalize(delta)),-1.0f,1.0f)));
            glm::mat4 segment = glm::translate(campus,(p+q)*.5f);
            if (glm::length(axis) > .0001f) segment = turned(segment,rotation,axis.x,axis.y,axis.z);
            r.draw(r.cylinder,scaled(segment,.035f,glm::length(delta),.035f),WHITE,{.57f,.77f,.79f});
        }
    }
}

void bench(Renderer& r, const glm::mat4& campus, float x, float z) {
    glm::mat4 seat = moved(campus,x,0,z);
    for (int i=0;i<4;++i)
        box(r,seat,{0,.55f,-.28f+i*.18f},{1.9f,.09f,.13f},WHITE,{.47f,.32f,.22f});
    box(r,seat,{0,.92f,-.39f},{1.9f,.48f,.10f},WHITE,{.47f,.32f,.22f});
    for (int side=-1;side<=1;side+=2) {
        box(r,seat,{side*.72f,.27f,0},{.10f,.54f,.10f},WHITE,{.31f,.34f,.32f});
        box(r,seat,{side*.72f,.70f,-.40f},{.09f,.70f,.09f},WHITE,{.31f,.34f,.32f});
    }
}

void lamp(Renderer& r, const glm::mat4& campus, float x, float z) {
    cylinder(r,campus,{x,1.65f,z},{.09f,3.3f,.09f},WHITE,{.34f,.38f,.37f});
    cylinder(r,campus,{x,3.35f,z},{.40f,.18f,.40f},CONCRETE,{.93f,.92f,.83f});
}

// 独立小巴模型沿地图路段移动；车体、车轮与轮毂保持层级变换。
void bus(Renderer& r, const glm::mat4& campus, const State& s) {
    ShuttlePose pose=r.shuttleRoute.pose(s.busDistance,r.heights);
    glm::mat4 vehicle=moved(campus,pose.position.x,pose.position.y,pose.position.z);
    vehicle=turned(turned(turned(vehicle,pose.yaw,0,1,0),pose.pitch,0,0,1),pose.roll,1,0,0);
    r.draw(r.shuttleBody,vehicle,WHITE,{.40f,.90f,.12f});
    r.draw(r.shuttleMetal,vehicle,WHITE,{.67f,.71f,.69f});
    r.draw(r.shuttleInterior,vehicle,WHITE,{.20f,.23f,.22f});
    r.draw(r.shuttleSeats,vehicle,WHITE,{.13f,.29f,.65f});
    r.draw(r.shuttleGlass,vehicle,GLASS,{.69f,.83f,.82f});
    r.draw(r.shuttleLamps,vehicle,WHITE,{.96f,.93f,.76f});
    r.draw(r.shuttleSign,vehicle,SHUTTLE_SIGN,glm::vec3(1));
    for (int x=-1;x<=1;x+=2) for (int z=-1;z<=1;z+=2) {
        glm::mat4 wheel=moved(vehicle,x*1.61f,.32f,z*.83f);
        wheel=turned(wheel,-glm::degrees(s.busDistance/.32f),0,0,1);
        glm::mat4 axial=turned(wheel,90,1,0,0);
        cylinder(r,axial,{0,0,0},{.64f,.20f,.64f},WHITE,{.10f,.11f,.11f});
        cylinder(r,axial,{0,0,0},{.39f,.215f,.39f},WHITE,{.55f,.57f,.55f});
        box(r,wheel,{0,0,z*.115f},{.045f,.29f,.018f},WHITE,{.25f,.27f,.26f});
        box(r,wheel,{0,0,z*.115f},{.29f,.045f,.018f},WHITE,{.25f,.27f,.26f});
    }
}

// 湖面上各个个体共享同一组独立模型，但使用错开的相位与各自的层级变换。
void wenshanSwans(Renderer& r, const glm::mat4& root, const State& s) {
    for (int i=0;i<5;++i) {
        SwanPose pose=r.swanRoute.pose(s.swanTime,i);
        glm::mat4 bird=turned(moved(root,pose.position.x,pose.position.y,pose.position.z),pose.yaw,0,1,0);
        if (!r.depthPass) r.draw(r.swanWake,(i==0 || i==4) ? bird : scaled(bird,.68f,1,.68f),WHITE,{.48f,.69f,.58f});
        if (i==0 || i==4) {
            r.draw(r.swanAdult,bird,SWAN_FEATHERS,{.17f,.19f,.19f},{1,1},5);
            r.draw(r.swanFeather,bird,SWAN_FEATHERS,{.21f,.23f,.23f},{1,1},5);
            r.draw(r.swanBill,bird,WHITE,{.84f,.13f,.10f});
            r.draw(r.swanBillTip,bird,WHITE,{.89f,.84f,.75f});
            r.draw(r.swanAdultEye,bird,WHITE,{.42f,.12f,.10f});
        } else {
            r.draw(r.swanChick,bird,SWAN_FEATHERS,{.62f,.60f,.57f},{1,1},5);
            r.draw(r.swanChickWing,bird,SWAN_FEATHERS,{.69f,.67f,.64f},{1,1},5);
            r.draw(r.swanChickBill,bird,WHITE,{.32f,.28f,.26f});
            r.draw(r.swanChickEye,bird,WHITE,{.10f,.09f,.08f});
        }
    }
}

// 凯风餐厅的壳体、楼梯、门廊和门扇分开绘制，以便独立调整与交互。
void kaifengCanteen(Renderer& r, const glm::mat4& root, const State& s) {
    r.draw(r.kaifengShell,root,KAIFENG_TILE,glm::vec3(1));
    r.draw(r.kaifengRoof,root,CONCRETE,{.88f,.88f,.84f});
    r.draw(r.kaifengInterior,root,PAVING,{.92f,.91f,.86f});
    r.draw(r.kaifengFurniture,root,WHITE,{.48f,.32f,.20f});
    r.draw(r.kaifengWindows,root,GLASS,{.61f,.79f,.72f});
    r.draw(r.kaifengYellow,root,KAIFENG_TILE,{1.0f,.75f,.16f});
    r.draw(r.kaifengRed,root,KAIFENG_TILE,{.73f,.18f,.16f});
    r.draw(r.kaifengStairs,root,CONCRETE,{.84f,.79f,.73f});
    r.draw(r.kaifengApron,root,PAVING,{.91f,.89f,.83f});
    r.draw(r.kaifengFrame,root,WHITE,{.92f,.92f,.88f});
    r.draw(r.kaifengRail,root,WHITE,{.64f,.68f,.66f});
    for (int side=0;side<2;++side) {
        glm::mat4 leaf=r.kaifengEntry.leaf(root,side!=0,s.kaifengDoor);
        r.draw(r.kaifengDoorGlass,leaf,GLASS,{.57f,.77f,.73f});
        r.draw(r.kaifengDoorFrame,leaf,WHITE,{.70f,.74f,.73f});
    }
}

// 立德门参考外国语学院门前照片及 2022 年航拍；细部和具体尺寸为估计。
void lideGate(Renderer& r, const glm::mat4& gate) {
    const float centerHeight=r.heights.at(-410,-54);
    for (int side=-1;side<=1;side+=2) {
        // 分别贴合门柱处的粗地形，避免一侧漂浮或被坡地埋没。
        float ground=r.heights.at(-410,-54+side*13.0f)-centerHeight;
        glm::mat4 pier=moved(gate,0,ground,side*13.0f);
        if (side < 0) {
            // 北侧门柱的两片立墙与中部横梁形成可辨识的开口。
            box(r,pier,{0,6.2f,-1.8f},{4.5f,12.4f,2.3f},FACADE_GATE,glm::vec3(1),{1,4});
            box(r,pier,{1.2f,5.7f,1.8f},{4.5f,11.4f,2.3f},FACADE_GATE,glm::vec3(1),{1,4});
            box(r,pier,{.6f,6.4f,0},{5.7f,1.2f,5.8f},FACADE_GATE,glm::vec3(1));
        } else {
            // 航拍可见右侧高墙前后错开；沿入口方向保留两道窄片。
            box(r,pier,{0,6.5f,-.9f},{4.5f,13.0f,4.4f},FACADE_GATE,glm::vec3(1),{1,4});
            box(r,pier,{2.1f,6.0f,2.0f},{4.5f,12.0f,3.0f},FACADE_GATE,glm::vec3(1),{1,4});
            box(r,pier,{-2.28f,1.75f,0},{.12f,2.9f,1.10f},GATE_SIGN,glm::vec3(1));
            box(r,pier,{-2.30f,2.50f,2.15f},{.13f,.48f,.95f},GATE_ADDRESS,glm::vec3(1));
        }
        // 岗亭以横向分缝墙、朝外玻璃门窗和悬挑檐口构成，而非实心盒体。
        box(r,pier,{3.3f,1.55f,side*4.3f},{7.6f,3.1f,7.0f},FACADE_GATE,{.80f,.82f,.79f},{1,1});
        box(r,pier,{2.2f,3.30f,side*3.5f},{10.0f,.45f,10.5f},CONCRETE,{.81f,.83f,.80f});
        box(r,pier,{2.2f,3.07f,side*3.5f},{9.8f,.12f,10.3f},WHITE,{.39f,.41f,.40f});
        box(r,pier,{-.56f,1.60f,side*4.5f},{.10f,2.05f,5.0f},GLASS,{.47f,.59f,.61f},{3,1});
        box(r,pier,{-.63f,1.55f,side*4.5f},{.10f,2.05f,.09f},WHITE,{.33f,.35f,.34f});
        box(r,pier,{-.63f,.58f,side*4.5f},{.10f,.08f,5.05f},WHITE,{.33f,.35f,.34f});
        box(r,pier,{3.45f,1.55f,side*.75f},{4.0f,2.0f,.10f},GLASS,{.48f,.60f,.62f},{2,1});
    }
    // 入口中央小型交通岛及抬起的道闸，反映航拍中的分流设施；位置和数量近似。
    float islandY=r.heights.at(-404.5f,-54)-centerHeight;
    glm::mat4 island=moved(gate,5.5f,islandY,0);
    box(r,island,{0,.15f,0},{6.8f,.30f,2.0f},CONCRETE,{.78f,.79f,.75f});
    box(r,island,{1.4f,.70f,0},{1.65f,1.1f,1.30f},WHITE,{.74f,.76f,.73f});
    box(r,island,{1.4f,1.45f,0},{1.4f,.9f,1.18f},GLASS,{.43f,.57f,.61f});
    box(r,island,{1.4f,1.97f,0},{1.8f,.15f,1.5f},CONCRETE,{.73f,.75f,.73f});
    for (int side=-1;side<=1;side+=2) {
        glm::mat4 barrier=moved(island,-1.5f,0,side*.78f);
        box(r,barrier,{0,.58f,0},{.45f,1.0f,.42f},WHITE,{.25f,.40f,.51f});
        box(r,barrier,{0,2.35f,0},{.10f,2.8f,.10f},WHITE,{.92f,.92f,.88f});
        for (int i=0;i<3;++i)
            box(r,barrier,{0,1.30f+i*.85f,0},{.115f,.36f,.115f},WHITE,{.75f,.10f,.10f});
    }
}

// 立功门 2020 年照片可见圆弧玻璃岗亭、浅色下墙和不锈钢闸机。
// 卫星影像及 OSM 仅能确定入口区域；设施位置、尺度和现状均为近似。
void legongGate(Renderer& r, const glm::mat4& campus) {
    const float boothX=300.8f, boothZ=-678.0f;
    glm::mat4 booth=moved(campus,boothX,r.heights.at(boothX,boothZ),boothZ);
    // 人行入口铺装分段贴合坡面，近似照片中的硬质地面并连接地图人行道。
    for (int i=0;i<6;++i) {
        float z=-682.0f+i*1.4f;
        box(r,campus,{298.9f,r.heights.at(298.9f,z)+.07f,z},{11.4f,.10f,1.41f},
            PAVING,{.85f,.86f,.81f},{2,1});
    }
    cylinder(r,booth,{0,.08f,0},{6.5f,.16f,6.5f},CONCRETE,{.78f,.80f,.77f});
    cylinder(r,booth,{0,.72f,0},{5.9f,1.28f,5.9f},WHITE,{.84f,.85f,.82f});
    cylinder(r,booth,{0,2.20f,0},{5.9f,1.68f,5.9f},GLASS,{.63f,.76f,.74f});
    const glm::vec3 frame(.24f,.31f,.31f);
    for (float y : {1.39f,2.05f,3.03f})
        cylinder(r,booth,{0,y,0},{6.0f,.075f,6.0f},WHITE,frame);
    for (int i=0;i<12;++i) {
        float angle=i*2.0f*PI/12.0f;
        box(r,booth,{2.95f*cos(angle),2.19f,2.95f*sin(angle)},
            {.085f,1.65f,.085f},WHITE,frame);
    }
    cylinder(r,booth,{0,3.14f,0},{6.45f,.18f,6.45f},CONCRETE,{.84f,.86f,.83f});
    // 仅在道路旁的人行入口示意闸机与栏杆；保留机动车道及环校路交叉口。
    for (int i=0;i<3;++i) {
        float x=294.0f+i*1.55f, z=-679.5f;
        glm::mat4 lane=moved(campus,x,r.heights.at(x,z),z);
        box(r,lane,{0,.52f,0},{.32f,1.04f,1.50f},WHITE,{.58f,.63f,.62f});
        box(r,lane,{0,1.06f,-.08f},{.36f,.08f,.52f},WHITE,{.23f,.31f,.30f});
        box(r,lane,{0,.65f,-.76f},{.21f,.47f,.025f},WHITE,{.22f,.53f,.39f});
        if (i<2) box(r,lane,{.77f,.84f,.35f},{1.25f,.055f,.055f},WHITE,{.71f,.75f,.73f});
    }
    // 门禁两旁的金属扶栏是照片可见元素，栏杆段数为近似。
    for (float x : {293.5f,297.7f}) {
        for (float z : {-681.0f,-678.0f})
            box(r,campus,{x,r.heights.at(x,z)+.56f,z},{.06f,1.12f,.06f},WHITE,{.64f,.69f,.67f});
        for (float y : {.58f,1.10f})
            box(r,campus,{x,r.heights.at(x,-679.5f)+y,-679.5f},{.06f,.06f,3.05f},WHITE,{.64f,.69f,.67f});
    }
    // 入口交叉处航拍可辨斑马线；宽度、条数及地面位置按影像近似。
    for (int i=0;i<9;++i) {
        float x=276.7f,z=-686.5f+i*1.15f;
        box(r,campus,{x,r.heights.at(x,z)+.035f,z},{4.2f,.035f,.50f},WHITE,{.91f,.91f,.87f});
    }
}

// 2024 年立言门照片可见机动车分流岛、岗亭及蓝色门禁设备；尺寸按道路和照片估计。
void liyanGate(Renderer& r, const glm::mat4& campus) {
    glm::mat4 gate=turned(moved(campus,137.0f,r.heights.at(137,337),337.0f),43.0f,0,1,0);
    box(r,gate,{0,.12f,-1.0f},{1.8f,.24f,7.0f},CONCRETE,{.85f,.85f,.79f});
    box(r,gate,{0,1.15f,-1.1f},{1.55f,2.05f,1.7f},GLASS,{.63f,.72f,.69f});
    box(r,gate,{0,.55f,-1.1f},{1.62f,.85f,1.76f},WHITE,{.82f,.84f,.79f});
    box(r,gate,{0,2.22f,-1.1f},{1.90f,.18f,2.0f},CONCRETE,{.90f,.90f,.85f});
    for (int side=-1;side<=1;side+=2) {
        glm::mat4 island=moved(gate,side*5.1f,0,0);
        box(r,island,{0,.11f,0},{1.0f,.22f,5.6f},CONCRETE,{.89f,.80f,.50f});
        for (int i=-2;i<=2;++i)
            box(r,island,{0,.225f,i*1.0f},{1.02f,.015f,.40f},WHITE,{.20f,.22f,.20f});
        box(r,island,{0,.73f,-.5f},{.54f,1.22f,.65f},WHITE,{.10f,.28f,.57f});
        cylinder(r,island,{0,1.75f,-1.15f},{.08f,2.5f,.08f},WHITE,{.42f,.44f,.42f});
        box(r,island,{0,2.98f,-1.15f},{.25f,.34f,.18f},WHITE,{.77f,.80f,.78f});
        // 抬起道闸以保留可探索通道，实际通行状态可能随时变化。
        box(r,island,{0,2.05f,.8f},{.09f,2.8f,.09f},WHITE,{.91f,.91f,.86f});
        for (int k=0;k<3;++k)
            box(r,island,{0,1.1f+k*.85f,.8f},{.11f,.37f,.11f},WHITE,{.74f,.12f,.10f});
    }
    // 入口西侧的斑马线参考卫星影像；条数、宽度和具体点位均为近似。
    for (int i=-5;i<=5;++i)
        box(r,gate,{i*1.05f,.035f,8.3f},{.55f,.035f,3.0f},WHITE,{.91f,.91f,.86f});
}

// 致远门前场和交通岛依据 OSM 与航拍表达，门禁设备仅做尺度近似。
void zhiyuanGate(Renderer& r, const glm::mat4& campus) {
    glm::mat4 gate=turned(moved(campus,192.9f,r.heights.at(192.9f,394.8f),394.8f),42.0f,0,1,0);
    // 细分石板铺装独立于大幅地形贴图，地面近景仍能看清分缝。
    box(r,gate,{0,.035f,-.3f},{33.5f,.05f,20.5f},PAVING,{.88f,.90f,.87f},{4,2.5f});
    for (int side=-1;side<=1;side+=2) {
        glm::mat4 island=moved(gate,side*4.1f,0,0);
        box(r,island,{0,.12f,0},{1.1f,.24f,5.8f},CONCRETE,{.84f,.85f,.80f});
        box(r,island,{0,.72f,-.5f},{.52f,1.20f,.62f},WHITE,{.18f,.34f,.53f});
        cylinder(r,island,{0,1.65f,1.0f},{.08f,2.5f,.08f},WHITE,{.48f,.50f,.48f});
        box(r,island,{0,2.90f,1.0f},{.24f,.30f,.20f},WHITE,{.83f,.85f,.81f});
    }
    // 学校 2026 年通知拟安装防撞升降柱；柱数和位置为示意，默认降至地面。
    for (int i=-3;i<=3;++i)
        cylinder(r,gate,{i*1.15f,.075f,5.0f},{.19f,.10f,.19f},WHITE,{.59f,.63f,.62f});
}

// 草坪前红心及白色校名由程序造型和字体重绘，位置、尺寸依据照片近似。
void zhiyuanLawnSign(Renderer& r, const glm::mat4& campus) {
    glm::mat4 sign=turned(moved(campus,210.0f,r.heights.at(210,421.5f),421.5f),220.0f,0,1,0);
    r.draw(r.cube,scaled(moved(sign,0,.24f,0),4.05f,.48f,.48f),ZHIYUAN_BASE,
           glm::vec3(1.2f),glm::vec2(1),4);
    r.draw(r.heart,scaled(moved(sign,-1.45f,.99f,0),.52f,.52f,1.0f),WHITE,{.98f,.07f,.15f},glm::vec2(1),4);
    for (float x : {-.47f,.20f,.87f,1.54f})
        box(r,sign,{x,.58f,0},{.045f,.22f,.05f},WHITE,{.84f,.86f,.83f});
    r.draw(r.signPlane,scaled(moved(sign,.52f,.99f,.05f),2.85f,.95f,1),ZHIYUAN_WORDS,
           glm::vec3(1.25f),glm::vec2(1),4);
}

// 星空广场“仰望星空”：四根回收灯柱及四色造型按校史照片近似。
void starPlaza(Renderer& r, const glm::mat4& plaza) {
    cylinder(r,plaza,{0,.16f,0},{11.2f,.32f,11.2f},STONE,{.83f,.84f,.80f});
    cylinder(r,plaza,{0,.34f,0},{8.6f,.18f,8.6f},CONCRETE,{.89f,.90f,.85f});
    const glm::vec3 colors[]={{.12f,.14f,.14f},{.75f,.10f,.10f},{.93f,.68f,.08f},{.09f,.30f,.68f}};
    for (int i=0;i<4;++i) {
        glm::mat4 column=moved(plaza,(i%2 ? .5f : -.5f),.42f,-3.6f+i*2.4f);
        cylinder(r,column,{0,2.4f,0},{.42f,4.8f,.42f},WHITE,colors[i]);
        r.draw(r.sphere,scaled(moved(column,0,5.2f,0),1.14f,1.55f,1.14f),WHITE,colors[i]);
        cylinder(r,column,{0,6.15f,0},{.25f,.7f,.25f},WHITE,colors[i]);
        if (i==0 || i==3) {
            glm::mat4 ring=turned(moved(column,0,6.8f,0),90,0,0,1);
            r.draw(r.torus,scaled(ring,.82f,.82f,.82f),WHITE,colors[i]);
        } else {
            r.draw(r.sphere,scaled(moved(column,0,6.75f,0),.55f,.78f,.55f),WHITE,colors[i]);
        }
    }
}

// 航拍中天人广场草坪有成组旗杆，数量与精确位置按照片近似。
void plazaFlags(Renderer& r, const glm::mat4& root) {
    const glm::vec3 colors[]={{.87f,.89f,.86f},{.48f,.61f,.71f},{.72f,.25f,.22f},{.92f,.78f,.53f}};
    for (int row=0;row<4;++row) for (int col=0;col<4;++col) {
        float x=-91.0f+row*5.8f, z=-5.5f-col*5.0f;
        glm::mat4 flag=moved(root,x,r.heights.at(x,z),z);
        cylinder(r,flag,{0,3,0},{.09f,6,.09f},WHITE,{.76f,.79f,.78f});
        box(r,flag,{.45f,5.55f,0},{.88f,.52f,.04f},WHITE,colors[(row+col)%4]);
    }
}

void world(Renderer& r, const State& s) {
    glm::mat4 root(1.0f);
    // 地形、道路、水系和建筑轮廓由开放地图与高程快照生成。
    r.draw(r.terrain,root,CAMPUS_GROUND,glm::vec3(1));
    r.draw(r.northCityGround,root,NORTH_CITY_GROUND,glm::vec3(1));
    // 细分道路与标线贴合真实地形三角面；低路缘沿路网外轮廓连续。
    r.draw(r.roadSurface,root,CAMPUS_ASPHALT,glm::vec3(1));
    r.draw(r.roadMarkings,root,WHITE,{.92f,.93f,.88f});
    r.draw(r.roadCurbs,root,CONCRETE,{.88f,.89f,.85f});
    // 人行道、地图步道和绿化分别是可独立维护的资源层，随地形起伏。
    r.draw(r.networkSidewalk,root,CAMPUS_SIDEWALK,glm::vec3(1));
    r.draw(r.networkPaths,root,PAVING,{.97f,.96f,.92f});
    // 立德门的砖面与边带使用独立网格，线位依据照片与门岗近似调整。
    r.draw(r.lideWalk,root,QIAOYUAN_BRICK,glm::vec3(1));
    r.draw(r.lideWalkEdge,root,CONCRETE,{.84f,.83f,.78f});
    // 航拍校准的草坪和步道分别贴地，Blender 导出的路缘与绿篱保留独立材质。
    r.draw(r.corridorPaving,root,CAMPUS_SIDEWALK,{1.0f,1.0f,.98f});
    r.draw(r.corridorLawn,root,GRASS,{1.05f,1.15f,.90f});
    r.draw(r.corridorBed,root,GRASS,{.54f,.60f,.39f});
    r.draw(r.corridorCurbs,root,CONCRETE,{.86f,.85f,.80f});
    r.draw(r.corridorLeaves,root,TREE_CLUSTER,{.74f,.91f,.68f},glm::vec2(1),2);
    // 时光广场砖面单独建模，遮挡旧底图中越过广场的道路纹理。
    r.draw(r.timePaving,root,CAMPUS_SIDEWALK,{1.0f,1.0f,.98f});
    // 路侧草带、矮草和灯具分层绘制，位置按可用空间近似而非实地测量。
    r.draw(r.networkVerges,root,GRASS,{.67f,.82f,.60f});
    r.draw(r.networkVergeGrass,root,WHITE,{.36f,.59f,.25f});
    r.draw(r.networkLamps,root,WHITE,{.69f,.73f,.70f});
    r.draw(r.networkLampHeads,root,WHITE,{.86f,.85f,.76f});
    r.draw(r.networkShrubs,root,TREE_CLUSTER,{.78f,.94f,.76f});
    r.draw(r.annotatedShrubs,root,TREE_CLUSTER,{.78f,.94f,.76f});
    r.draw(r.networkStairs,root,PAVING,{.95f,.94f,.90f});
    r.draw(r.dujuanTrail,root,PAVING,{.89f,.87f,.81f});
    r.draw(r.dujuanBoardwalk,root,DUJUAN_BOARDWALK,{1,1,1});
    // 乔相阁前的园路、校巴造型与西苑街面均随局部坡地生成。
    r.draw(r.gardenBrick,root,QIAOYUAN_BRICK,glm::vec3(1));
    r.draw(r.gardenWalk,root,QIAOYUAN_GARDEN_WALK,glm::vec3(1));
    r.draw(r.gardenMulch,root,WHITE,{.43f,.34f,.25f});
    r.draw(r.gardenFrame,root,WHITE,{.56f,.80f,.78f});
    r.draw(r.gardenWords,root,QIAOYUAN_GARDEN_WORDS,glm::vec3(1));
    r.draw(r.gardenWood,root,WHITE,{.47f,.32f,.21f});
    r.draw(r.gardenWoodTops,root,WHITE,{.68f,.51f,.35f});
    r.draw(r.gardenPlants,root,WHITE,{.29f,.49f,.26f});
    r.draw(r.gardenPlantsLight,root,WHITE,{.48f,.65f,.32f});
    r.draw(r.gardenStones,root,STONE,{.87f,.84f,.76f});
    r.draw(r.shopPaving,root,XIYUAN_PAVING,glm::vec3(1));
    r.draw(r.shopOppositePaving,root,PAVING,{.91f,.91f,.88f});
    r.draw(r.shopRedPaving,root,QIAOYUAN_BRICK,{1.05f,.86f,.83f});
    r.draw(r.shopPlantBed,root,WHITE,{.32f,.29f,.22f});
    r.draw(r.shopPlantLeaves,root,WHITE,{.34f,.59f,.27f});
    if (!r.depthPass) r.draw(r.campusWater,root,WATER,{.85f,1.00f,.68f},{1,1},1);
    wenshanSwans(r,root,s);
    r.draw(r.waterBank,root,STONE,{.87f,.88f,.82f});
    r.draw(r.wenshanTerraces,root,STONE,{.94f,.95f,.91f});
    r.draw(r.wenshanPaving,root,STONE,{.65f,.67f,.63f});
    r.draw(r.wenshanAncientStone,root,WENSHAN_ANCIENT_STONE,glm::vec3(1));
    r.draw(r.wenshanAncientStonePlanting,root,TREE_CLUSTER,{.78f,.94f,.76f});
    kaifengCanteen(r,root,s);
    // 师范楼与北夏馆旁临坡基础为独立网格，按地形近似接地。
    r.draw(r.teachersFoundation,root,CONCRETE,{.69f,.70f,.67f});
    r.draw(r.northsportsFoundation,root,CONCRETE,{.69f,.70f,.67f});
    r.draw(r.southdormFoundation,root,CONCRETE,{.69f,.70f,.67f});
    r.draw(r.windzhaiFoundation,root,CONCRETE,{.69f,.70f,.67f});
    r.draw(r.huiwenroadFoundation,root,CONCRETE,{.69f,.70f,.67f});
    r.draw(r.facultyroadFoundation,root,CONCRETE,{.69f,.70f,.67f});
    r.draw(r.wallCream,root,FACADE_CREAM,glm::vec3(1));
    r.draw(r.wallRed,root,FACADE_RED,glm::vec3(1));
    r.draw(r.xiyuanWall,root,XIYUAN_FACADE,glm::vec3(1));
    r.draw(r.xiyuanFrame,root,WHITE,{.96f,.95f,.90f});
    r.draw(r.xiyuanRailing,root,WHITE,{.36f,.39f,.37f});
    r.draw(r.shopShell,root,CONCRETE,{.90f,.84f,.72f});
    r.draw(r.shopRoof,root,CONCRETE,{.86f,.86f,.82f});
    r.draw(r.shopMannerWall,root,MANNER_WALL,glm::vec3(1));
    r.draw(r.shopGlass,root,XIYUAN_WINDOW,glm::vec3(1));
    r.draw(r.shopBaseDark,root,WHITE,{.24f,.25f,.23f});
    r.draw(r.shopBaseBrick,root,QIAOYUAN_BRICK,glm::vec3(1));
    r.draw(r.shopFrames,root,WHITE,{.36f,.38f,.36f});
    r.draw(r.shopCanopy,root,WHITE,{.88f,.88f,.82f});
    r.draw(r.shopMembrane,root,WHITE,{.92f,.91f,.84f});
    r.draw(r.shopAwningFrame,root,WHITE,{.95f,.95f,.91f});
    r.draw(r.shopSigns,root,XIYUAN_SIGNS,glm::vec3(1));
    r.draw(r.shopBicycle,root,WHITE,{.20f,.23f,.22f});
    r.draw(r.wallHuiyan,root,FACADE_HUIYAN,glm::vec3(1));
    r.draw(r.wallCity,root,FACADE_CITY,glm::vec3(1));
    r.draw(r.wallStrip,root,FACADE_STRIP,glm::vec3(1));
    // 汇紫楼底层的柱廊和退后的暗色界面是独立几何，不只是立面贴图。
    r.draw(r.huiziShell,root,WHITE,{1.01f,1.01f,.98f},{1,1},4);
    r.draw(r.huiziGround,root,FACADE_ATRIUM,{.48f,.50f,.49f});
    r.draw(r.huiziCanopy,root,FACADE_ATRIUM,{.62f,.72f,.78f});
    r.draw(r.huiziPaving,root,CONCRETE,{1.0f,.99f,.94f});
    // 汇紫楼屋面及设备根据航拍近似，均使用程序生成的材质。
    r.draw(r.huiziRoof,root,HUIZI_ROOF,glm::vec3(1));
    r.draw(r.huiziEquipment,root,HUIZI_EQUIPMENT,glm::vec3(1));
    // 汇元楼的石质门廊、校名和穿行开口对照照片和地图人行线近似。
    r.draw(r.huiyuanShell,root,CONCRETE,{1.0f,1.0f,.98f});
    r.draw(r.huiyuanGround,root,FACADE_ATRIUM,{.48f,.50f,.49f});
    r.draw(r.huiyuanSign,root,HUIYUAN_SIGN,glm::vec3(1));
    r.draw(r.huiyuanMotto,root,HUIYUAN_MOTTO,glm::vec3(1));
    r.draw(r.huiyuanPaving,root,CONCRETE,{1.0f,.99f,.95f});
    r.draw(r.huiyuanPlanting,root,GRASS,{.82f,1.0f,.75f});
    r.draw(r.huiyuanEntryRoof,root,WHITE,{.56f,.24f,.20f});
    // 演会中心保留开放侧面、红色屋架和观众席，不将厅体封成玻璃盒子。
    r.draw(r.yanhuiRoof,root,WHITE,{.97f,.98f,.95f});
    r.draw(r.yanhuiSeams,root,WHITE,{.49f,.50f,.47f});
    r.draw(r.yanhuiUnder,root,WHITE,{.36f,.36f,.34f});
    r.draw(r.yanhuiSteel,root,WHITE,{.46f,.13f,.11f});
    r.draw(r.yanhuiStone,root,STONE,{.86f,.84f,.79f});
    r.draw(r.yanhuiFloor,root,CONCRETE,{.90f,.90f,.85f});
    r.draw(r.yanhuiSeats,root,WHITE,{1.05f,.34f,.04f},{1,1},4);
    r.draw(r.yanhuiSeatDetail,root,WHITE,{1.30f,.57f,.07f},{1,1},4);
    r.draw(r.yanhuiStage,root,WHITE,{.72f,.47f,.25f});
    r.draw(r.yanhuiScreen,root,WHITE,{.80f,.82f,.86f});
    r.draw(r.yanhuiSteps,root,STONE,{.94f,.92f,.86f});
    r.draw(r.yanhuiPineTrunks,root,STONE,{.44f,.37f,.26f});
    r.draw(r.yanhuiPineCrowns,root,WHITE,{.18f,.31f,.13f},{1,1},2);
    // 汇文楼根据航拍和入口照片区分格窗、悬跨门洞、红墙与半圆退台。
    r.draw(r.wallHuiwen,root,FACADE_HUIWEN,glm::vec3(1));
    r.draw(r.huiwenDetail,root,CONCRETE,{.94f,.94f,.90f});
    r.draw(r.huiwenTerrace,root,WHITE,{.55f,.46f,.35f});
    r.draw(r.huiwenRed,root,FACADE_HUIWEN_RED,glm::vec3(1));
    r.draw(r.huiwenSteps,root,STONE,{.88f,.88f,.83f});
    r.draw(r.wallPerforated,root,FACADE_PERFORATED,glm::vec3(1));
    // 致艺楼厚墙、内凹窗和设备格栅分材质绘制，使窗洞受光照影响。
    r.draw(r.zhiyiShell,root,WHITE,{1.10f,1.10f,1.07f},{1,1},4);
    r.draw(r.zhiyiGlass,root,FACADE_ATRIUM,{.80f,.88f,.87f});
    r.draw(r.zhiyiVents,root,FACADE_ZHIYI_VENT,{1.10f,1.10f,1.07f},{1,1},4);
    r.draw(r.zhiyiRails,root,WHITE,{.48f,.52f,.51f});
    r.draw(r.zhiyiPodium,root,FACADE_ZHIYI_PODIUM,{1.10f,1.10f,1.07f},{1,1},4);
    r.draw(r.zhiyiSkywalk,root,WHITE,{1.08f,1.08f,1.04f},{1,1},4);
    r.draw(r.zhiyiSkyRails,root,WHITE,{.45f,.48f,.46f});
    // 致真楼采用真正内凹的窗洞、白色立体格架与深灰短侧，演播厅单独绘制。
    r.draw(r.zhizhenShell,root,WHITE,{1.05f,1.05f,1.03f},{1,1},4);
    r.draw(r.zhizhenGlass,root,FACADE_ATRIUM,{.78f,.84f,.83f});
    r.draw(r.zhizhenSide,root,WHITE,{.55f,.56f,.55f});
    r.draw(r.zhizhenFrame,root,WHITE,{1.10f,1.10f,1.07f},{1,1},4);
    r.draw(r.zhizhenPodium,root,FACADE_ATRIUM,{.80f,.84f,.83f});
    // 致知楼白色窗格及 L4 深色横窗和屋顶绿化按航拍近似。
    r.draw(r.zhizhiShell,root,WHITE,{1.04f,1.04f,1.01f},{1,1},4);
    r.draw(r.zhizhiGlass,root,GLASS,{.85f,.94f,.94f});
    r.draw(r.zhizhiFrame,root,WHITE,{1.03f,1.03f,1.0f},{1,1},4);
    r.draw(r.l4Glass,root,GLASS,{.40f,.47f,.48f});
    r.draw(r.l4Frame,root,WHITE,{.68f,.71f,.68f});
    r.draw(r.l4Garden,root,GRASS,{.70f,.81f,.56f});
    r.draw(r.l4Equipment,root,CONCRETE,{.84f,.85f,.80f});
    // 致工楼门廊、内凹窗与停车场均参考校方照片和地图重建。
    r.draw(r.zhigongShell,root,WHITE,{1.04f,1.04f,1.01f},{1,1},4);
    r.draw(r.zhigongGlass,root,GLASS,{.76f,.86f,.85f});
    r.draw(r.zhigongDark,root,WHITE,{.47f,.51f,.50f});
    r.draw(r.zhigongPerforated,root,FACADE_ZHIYI_VENT,{1,1,1},{1,1},4);
    r.draw(r.zhigongLouvers,root,ZHIGONG_LOUVER,{1,1,1});
    r.draw(r.zhigongRails,root,WHITE,{.46f,.49f,.48f});
    r.draw(r.zhigongSteps,root,CONCRETE,{.92f,.92f,.89f},{1,1},4);
    r.draw(r.zhigongSign,root,ZHIGONG_SIGN,{1,1,1},{1,1},4);
    r.draw(r.zhigongCarsLight,root,WHITE,{.92f,.93f,.91f});
    r.draw(r.zhigongCarsDark,root,WHITE,{.40f,.44f,.46f});
    r.draw(r.zhigongCarsGlass,root,GLASS,{.56f,.66f,.69f});
    r.draw(r.zhigongCarsWheels,root,WHITE,{.20f,.21f,.20f});
    r.draw(r.zhigongChargers,root,WHITE,{.90f,.92f,.90f});
    r.draw(r.studioShell,root,WHITE,{.31f,.32f,.33f});
    r.draw(r.studioGlass,root,FACADE_ATRIUM,{.76f,.80f,.79f});
    r.draw(r.studioSign,root,STUDIO_SIGN,glm::vec3(1),{1,1},4);
    r.draw(r.studioColumns,root,WHITE,{1.02f,1.02f,1.0f},{1,1},4);
    r.draw(r.zhiliGlass,root,FACADE_ATRIUM,{.82f,.86f,.86f});
    r.draw(r.zhiliFrame,root,WHITE,{.98f,.98f,.96f},{1,1},4);
    r.draw(r.zhiliSide,root,FACADE_ZHILI_SIDE,{1.0f,1.0f,.98f},{1,1},4);
    // 致理楼 L1 的水平窗带、入口及连接 L3 的多层步行桥。
    r.draw(r.zhiliAShell,root,WHITE,{1.03f,1.03f,1.0f},{1,1},4);
    r.draw(r.zhiliAGlass,root,GLASS,{.69f,.80f,.82f});
    r.draw(r.zhiliAFrame,root,WHITE,{.48f,.52f,.52f});
    r.draw(r.zhiliASign,root,ZHILI_A_SIGN,glm::vec3(1),{1,1},4);
    r.draw(r.zhiliHighWalkways,root,WHITE,{.90f,.91f,.88f},{1,1},4);
    r.draw(r.zhiliHighRails,root,WHITE,{.48f,.52f,.51f});
    // 地图和实景中的高层悬挑体有厚白框与内退玻璃，不用贴图伪装为平面。
    r.draw(r.zhiliCrown,root,WHITE,{1.10f,1.10f,1.07f},{1,1},4);
    r.draw(r.zhiliCrownGlass,root,FACADE_ATRIUM,{.79f,.88f,.88f});
    r.draw(r.zhiliCrownFrame,root,WHITE,{1.05f,1.05f,1.02f},{1,1},4);
    // 汇萃楼根据总平面图区分三层基座、L 形玻璃高层与屋顶舞台体量。
    r.draw(r.huicuiShell,root,WHITE,{1.05f,1.05f,1.03f},{1,1},4);
    r.draw(r.huicuiGlass,root,FACADE_ATRIUM,{.84f,.91f,.94f});
    r.draw(r.huicuiUpper,root,FACADE_ATRIUM,{.74f,.83f,.90f});
    r.draw(r.huicuiStage,root,WHITE,{1.03f,1.03f,1.01f},{1,1},4);
    r.draw(r.huicuiFrames,root,WHITE,{1.06f,1.06f,1.04f},{1,1},4);
    r.draw(r.huicuiSkylights,root,FACADE_ATRIUM,{.76f,.84f,.89f});
    r.draw(r.wallTeng,root,FACADE_TENG,glm::vec3(1));
    r.draw(r.wallZhai,root,FACADE_ZHAI,glm::vec3(1));
    r.draw(r.zhaiStone,root,CONCRETE,{.48f,.49f,.48f});
    r.draw(r.zhaiPaving,root,CAMPUS_SIDEWALK,{.90f,.91f,.89f});
    r.draw(r.landmarkGlass,root,FACADE_ATRIUM,glm::vec3(1));
    r.draw(r.roofs,root,CONCRETE,{.91f,.92f,.89f},{1,1});
    // 南区饭堂按照片表达密集白色格栅、红色楼层牌及二层外廊。
    r.draw(r.canteenGlass,root,FACADE_ATRIUM,{1.12f,1.20f,1.22f});
    r.draw(r.canteenFrame,root,WHITE,{1.04f,1.04f,1.02f},{1,1},4);
    r.draw(r.canteenRoof,root,HUIZI_ROOF,{.82f,.83f,.80f});
    r.draw(r.canteenSign,root,CANTEEN_SIGN,glm::vec3(1),{1,1},4);
    r.draw(r.archiveWall,root,CONCRETE,{.92f,.93f,.89f});
    r.draw(r.archiveSeal,root,ARCHIVE_SEAL,glm::vec3(1));
    r.draw(r.archiveFins,root,WHITE,{.76f,.79f,.76f});
    r.draw(r.archiveDoor,root,FACADE_ATRIUM,{.85f,.90f,.88f});
    r.draw(r.bridge,root,CONCRETE,{.76f,.78f,.75f},{1,1});
    r.draw(r.bridgeRoad,root,WHITE,{.33f,.36f,.35f});
    r.draw(r.bridgeMarkings,root,WHITE,{.85f,.84f,.77f});
    r.draw(r.zhounanYellow,root,WHITE,{.92f,.68f,.10f});
    r.draw(r.bridgeSupport,root,CONCRETE,{.75f,.78f,.76f});
    // 周南桥主跨用双层桥板及钢管斜撑表达可见结构，标高与管径为估计。
    r.draw(r.bridgeLower,root,CONCRETE,{.76f,.78f,.76f});
    r.draw(r.zhounanZhaiTreads,root,ZHOUNAN_ZHAI_TILE,glm::vec3(1));
    r.draw(r.zhounanZhaiWalls,root,CONCRETE,{.79f,.82f,.81f});
    // 桥下为受遮挡的浅色混凝土板，适当提高环境光以保留横梁和空间层次。
    r.draw(r.zhounanCeiling,root,WHITE,{.94f,.95f,.92f},{1,1},4);
    r.draw(r.bridgeTruss,root,WHITE,{.67f,.71f,.70f});
    r.draw(r.zhounanPots,root,WHITE,{.91f,.91f,.87f});
    r.draw(r.zhounanPlants,root,WHITE,{.26f,.48f,.16f},{1,1},2);
    r.draw(r.bridgeRoof,root,CONCRETE,{.92f,.94f,.91f});
    r.draw(r.bridgeRails,root,WHITE,{.78f,.82f,.80f},{1,1});
    // 跑道与球场使用独立高分辨率纹理；球门保留立体几何。
    r.draw(r.stadiumSurface,root,STADIUM_SURFACE,glm::vec3(1));
    r.draw(r.stadiumGoals,root,WHITE,{.91f,.92f,.87f});
    r.draw(r.stadiumEntryPaving,root,STADIUM_ENTRY_PAVING,glm::vec3(1));
    r.draw(r.stadiumEntryEdges,root,CONCRETE,{.83f,.83f,.80f});
    r.draw(r.stadiumEntryMetal,root,WHITE,{.83f,.85f,.83f});
    r.draw(r.standBase,root,CONCRETE,{.83f,.84f,.81f});
    r.draw(r.redStands,root,STADIUM_SEAT,{.95f,.24f,.18f});
    r.draw(r.blueStands,root,STADIUM_SEAT,{.20f,.45f,.70f});
    r.draw(r.orangeStands,root,STADIUM_SEAT,{1.02f,.43f,.14f});
    r.draw(r.yellowStands,root,STADIUM_SEAT,{1.05f,.82f,.15f});
    r.draw(r.standLines,root,WHITE,{.83f,.84f,.80f});
    r.draw(r.stadiumCanopy,root,WHITE,{.95f,.96f,.92f});
    r.draw(r.stadiumMasts,root,WHITE,{.83f,.85f,.83f});
    r.draw(r.stadiumLamps,root,WHITE,{.97f,.94f,.86f});
    // 运动广场与至柔馆的球场位于高架平台；两处高度为资料约束下的估计。
    r.draw(r.sportsShell,root,CONCRETE,{.78f,.80f,.77f});
    r.draw(r.sportsRoof,root,CONCRETE,{.84f,.85f,.82f});
    r.draw(r.basketballCourt,root,BASKETBALL_COURT,glm::vec3(1));
    r.draw(r.tennisCourt,root,TENNIS_COURT,glm::vec3(1));
    r.draw(r.sportsRails,root,WHITE,{.76f,.80f,.78f});
    // 依据照片概括篮架、围网与照明；网面保留透明孔洞，不遮挡场地。
    r.draw(r.basketballBoards,root,WHITE,{.79f,.87f,.86f});
    r.draw(r.basketballRims,root,WHITE,{.76f,.24f,.13f});
    r.draw(r.sportsNetting,root,SPORTS_FENCE,glm::vec3(1));
    // 石头坞及南秋馆依据 2026 年实景制作，平台高差与局部构造为近似。
    r.draw(r.wallFortress,root,FACADE_FORTRESS,glm::vec3(1));
    // 学活两翼基础与坡地相接；中庭不使用该基础墙封闭。
    r.draw(r.activityFoundation,root,CONCRETE,{.67f,.68f,.65f});
    r.draw(r.wallActivity,root,FACADE_ACTIVITY,glm::vec3(1));
    r.draw(r.stoneShell,root,CONCRETE,{.78f,.79f,.75f});
    r.draw(r.stoneFlat,root,CONCRETE,{.91f,.92f,.88f});
    r.draw(r.stonePlaza,root,STONE_PLAZA,glm::vec3(1));
    r.draw(r.southPool,root,SOUTH_POOL,glm::vec3(1),{1,1},1);
    r.draw(r.stoneCanopies,root,WHITE,{.95f,.96f,.91f});
    r.draw(r.stoneRails,root,WHITE,{.90f,.92f,.88f});
    // 元平体育馆的门廊、桁架和浮雕，以及北夏馆的两个露天泳池。
    r.draw(r.yuanpingShell,root,WHITE,{.90f,.91f,.88f});
    r.draw(r.yuanpingBase,root,FACADE_STRIP,{.72f,.75f,.74f});
    r.draw(r.yuanpingFoundation,root,CONCRETE,{.69f,.70f,.67f});
    r.draw(r.yuanpingRoof,root,WHITE,{.95f,.96f,.94f});
    r.draw(r.yuanpingTruss,root,WHITE,{.62f,.65f,.64f});
    r.draw(r.yuanpingRelief,root,YUANPING_RELIEF,glm::vec3(1));
    r.draw(r.yuanpingSeal,root,YUANPING_SEAL,glm::vec3(1));
    r.draw(r.yuanpingGlass,root,GLASS,{.55f,.65f,.68f});
    r.draw(r.yuanpingSteps,root,CONCRETE,{.87f,.88f,.84f});
    r.draw(r.yuanpingName,root,YUANPING_NAME,glm::vec3(1));
    r.draw(r.yuanpingGarden,root,WHITE,{.34f,.48f,.22f});
    r.draw(r.yuanpingEntryPaving,root,YUANPING_ENTRY_PAVING,glm::vec3(1));
    r.draw(r.beixiaShell,root,WHITE,{.85f,.87f,.85f});
    r.draw(r.beixiaRoof,root,WHITE,{.25f,.54f,.69f});
    r.draw(r.beixiaUnder,root,WHITE,{.35f,.42f,.43f});
    r.draw(r.beixiaFrame,root,WHITE,{.45f,.52f,.53f});
    r.draw(r.beixiaSeats,root,WHITE,{.32f,.57f,.75f});
    r.draw(r.beixiaDeck,root,CONCRETE,{.93f,.93f,.89f});
    r.draw(r.beixiaLanes,root,SOUTH_POOL,glm::vec3(1),{1,1},1);
    r.draw(r.beixiaPractice,root,WHITE,{.28f,.71f,.81f});
    r.draw(r.beixiaCoping,root,WHITE,{.93f,.94f,.89f});
    // 高尔夫场弧形击球棚与距离牌按地图及照片近似。
    r.draw(r.golfFloor,root,CONCRETE,{.83f,.84f,.79f});
    r.draw(r.golfRoof,root,WHITE,{.91f,.91f,.86f});
    r.draw(r.golfFrame,root,WHITE,{.67f,.70f,.66f});
    r.draw(r.golfMats,root,WHITE,{.17f,.38f,.17f});
    r.draw(r.golfSigns,root,GOLF_DISTANCE,glm::vec3(1));
    // 时光广场日晷按地图点位及校方照片概括，盘面倾角和尺寸为估计。
    r.draw(r.sundialStone,root,SUNDIAL_STONE,glm::vec3(1));
    r.draw(r.sundialFace,root,SUNDIAL_FACE,glm::vec3(1));
    r.draw(r.sundialPointer,root,WHITE,{.62f,.65f,.62f});
    r.draw(r.timeCurb,root,CONCRETE,{.94f,.94f,.90f});
    r.draw(r.timeHedges,root,WHITE,{.31f,.45f,.19f});
    r.draw(r.timeHedgeLeaves,root,TREE_CLUSTER,{.73f,.89f,.67f});
    r.draw(r.alumniGarden,root,GRASS,{.87f,.96f,.80f});

    // 三处标志建筑单独建模；位置由对应 OSM 建筑轮廓定位。
    glm::mat4 north = turned(moved(root,-126.4f,r.heights.at(-126.4f,-102.9f),-102.9f),11,0,1,0);
    northLibrary(r,north);
    r.draw(r.northEntrySteps,root,CONCRETE,{.87f,.88f,.84f});
    r.draw(r.northEntryPaving,root,NORTH_LIBRARY_ENTRY_PAVING,glm::vec3(1));
    r.draw(r.northEntryEdges,root,STONE,{.83f,.85f,.82f});
    r.draw(r.northUpperDeck,root,CONCRETE,{.94f,.96f,.94f});
    r.draw(r.northUpperStairs,root,CONCRETE,{.94f,.96f,.94f});
    r.draw(r.northUpperRails,root,WHITE,{.72f,.75f,.73f});
    r.draw(r.northEntryFrames,root,WHITE,{.43f,.50f,.52f});
    glm::mat4 south = turned(moved(root,-62.4f,r.heights.at(-62.4f,128.1f),128.1f),11,0,1,0);
    southLibrary(r,scaled(south,2.0f,1.42f,2.4f));
    glm::mat4 science = moved(root,-12.3f,r.heights.at(-12.3f,-300.3f),-300.3f);
    huixing(r,science);
    // 北侧台阶沿地图线位，规则绿地及低矮灌木参考航拍。
    r.draw(r.huixingSteps,root,STONE,{.87f,.88f,.82f});
    r.draw(r.huixingHedges,root,WHITE,{.34f,.48f,.25f});
    r.gates.draw(r.depthPass ? r.depth : r.scene,r.depthPass,root);
    if (r.lideAsset) r.lideAsset->draw(r.depthPass ? r.depth : r.scene,r.depthPass,root,s.lideAngle);
    r.qiaoyuanDorms.draw(r.depthPass ? r.depth : r.scene,r.depthPass,root);
    r.southDorms.draw(r.depthPass ? r.depth : r.scene,r.depthPass,root);
    r.southSpine.draw(r.depthPass ? r.depth : r.scene,r.depthPass,root);
    zhiyuanLawnSign(r,root);
    starPlaza(r,moved(root,291,r.heights.at(291,-400),-400));
    plazaFlags(r,root);

    // 两馆之间的“天地人和”雕塑及可交互的小巴保留层级变换。
    glm::mat4 plaza = moved(root,-108,r.heights.at(-108,36)+.15f,36);
    cylinder(r,plaza,{0,.06f,0},{16,.12f,16},PAVING);
    for (int i=0;i<64;++i) {
        glm::mat4 ray=turned(plaza,i*360.0f/64.0f,0,1,0);
        box(r,ray,{5.55f,.14f,0},{4.15f,.025f,.055f},WHITE,{.56f,.59f,.55f});
    }
    armillary(r,plaza,s);
    for (int i=0;i<5;++i)
        palm(r,plaza,-22.0f+i*8.0f,-15.0f,6.5f+(i%2)*.6f,i);
    bench(r,plaza,14,8); bench(r,plaza,18,8);
    lamp(r,plaza,-7,-14); lamp(r,plaza,7,-14);
    bus(r,root,s);
    r.trees();
    // 北门入校路两侧有棕榈列植的文字记载；数量和逐棵点位按道路近似。
    const float legongPalmZ[]={-668,-648,-628,-608,-588,-568,-548,-488,-468};
    for (int i=0;i<9;++i) for (int side=-1;side<=1;side+=2) {
        float z=legongPalmZ[i], x=286.0f+(z+688.0f)*.019f+side*12.5f;
        royalPalm(r,root,x,z,9.4f+(i%3)*.65f,i*2+(side>0));
    }
    const float palmX[]={-107,-102,-97,-92,-63,-57,-51,-45,-39,-33};
    for (int i=0;i<10;++i) {
        float z=109.0f-.194f*(palmX[i]+65.0f);
        royalPalm(r,root,palmX[i],z,9.0f+(i%4)*.7f,i);
    }
    // 北馆入口两侧的棕榈群参照航拍和入口照片，具体株数、点位未实测。
    const glm::vec2 northPalm[]={{-142,-62},{-146,-56},{-97,-71},{-92,-66}};
    for (int i=0;i<4;++i) royalPalm(r,root,northPalm[i].x,northPalm[i].y,9.5f+(i%2)*1.2f,i+10);
    // 教学楼旧照中平台一端可见棕榈，位置及株数按照片概括。
    const glm::vec2 teachingPalm[]={{-25,45},{-27,47},{-23,48}};
    for (int i=0;i<3;++i) {
        glm::mat4 base=moved(root,0,r.heights.at(teachingPalm[i].x,teachingPalm[i].y),0);
        palm(r,base,teachingPalm[i].x,teachingPalm[i].y,6.2f+i*.35f,i+20);
    }
    // 场馆门廊两侧照片可见种植池与热带植物；株数及位置按照片近似。
    const glm::vec2 gymPalms[]={{250.8f,-368.6f},{249.7f,-372.5f},{240.6f,-403.2f},{239.5f,-407.0f}};
    glm::mat4 gymPlantBase=moved(root,0,r.heights.at(204.1f,-373.8f)-.1f,0);
    for (int i=0;i<4;++i) palm(r,gymPlantBase,gymPalms[i].x,gymPalms[i].y,4.3f+(i%2)*.4f,i+30);
    // 汇元楼校名旁的低矮树冠以简单几何近似，具体树形和位置未经测量。
    glm::mat4 officePlant=moved(root,-224.3f,r.heights.at(-224.3f,-16.0f)+.68f,-16.0f);
    cylinder(r,officePlant,{0,.7f,0},{.25f,1.4f,.25f},STONE,{.48f,.40f,.29f});
    const glm::vec3 officeCrown[]={{-1.0f,1.65f,0},{.2f,1.95f,.2f},{1.0f,1.5f,-.2f}};
    for (int i=0;i<3;++i)
        r.draw(r.sphere,scaled(moved(officePlant,officeCrown[i].x,officeCrown[i].y,officeCrown[i].z),
                               1.7f,.55f,1.15f),WHITE,{.25f,.42f,.20f},glm::vec2(1),2);
    // 上文山湖西岸实景中的树木与小型热带植物；具体株型和尺度按照片近似。
    float lakeY=r.heights.at(-377,190);
    glm::mat4 lakePlanting=moved(root,0,lakeY+.91f,0);
    tree(r,lakePlanting,-381.5f,185.5f,2.05f,5);
    for (int i=0;i<6;++i) {
        float a=glm::radians(-62.5f+i*25.0f);
        float forward=8.4f*cos(a), side=8.4f*sin(a);
        glm::mat4 lowPlant=turned(moved(lakePlanting,-377+.72f*forward-.69f*side,0,
                                          190+.69f*forward+.72f*side),i*27.0f,0,1,0);
        r.draw(r.palmMesh,scaled(lowPlant,.62f,.48f+(i%2)*.04f,.62f),PALM_ATLAS,glm::vec3(1));
    }
    const glm::vec2 garden[]={{-420,192},{-413,187},{-406,182}};
    for (int i=0;i<3;++i) {
        glm::vec3 color=i==1 ? glm::vec3(.76f,.92f,.66f) : glm::vec3(.76f,.54f,.66f);
        r.draw(r.sphere,scaled(moved(root,garden[i].x,lakeY+1.58f,garden[i].y),
                                2.2f,1.05f,2.2f),TREE_CANOPY,color,glm::vec2(1),2);
    }
    // 东南岸观景台的低矮石坐凳按照片示意，具体数量与位置未经测量。
    for (int i=-1;i<=1;++i)
        box(r,root,{-227.5f,lakeY+1.13f,211.0f+i*6.0f},{.62f,.42f,3.6f},STONE,{.85f,.86f,.82f});
    if (!r.depthPass) {
        // 召南桥玻璃护栏不写入深度，也不投射不透明阴影。
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
        glDepthMask(GL_FALSE);
        r.draw(r.bridgeGlass,root,WHITE,{.66f,.85f,.83f},{1,1},3);
        // 南馆中庭上方两层连廊的玻璃栏板，结构位置按公开照片近似。
        glm::mat4 southBuilding=turned(scaled(south,2.0f,1.42f,2.4f),180,0,1,0);
        for (float level : {8.0f,13.55f}) for (int side=-1;side<=1;side+=2)
            r.draw(r.cube,scaled(moved(southBuilding,7.05f,level+.52f,side*3.48f),5.15f,.92f,.055f),
                   GLASS,{.78f,.94f,.90f},{1,1},3);
        // 汇星楼屋顶交通筒的玻璃包围可透见内部实体和已绘制的细金属框架。
        r.draw(r.cube,scaled(moved(science,-.51f,83.75f,1.53f),12.44f,22.5f,24.17f),
               GLASS,{.73f,.91f,.89f},{1,1},3);
        glDepthMask(GL_TRUE);
        glDisable(GL_BLEND);
    }
}

void render(Renderer& r, const State& state, int width, int height, float seconds) {
    glm::vec3 sun = glm::normalize(glm::vec3(-.40f,1.0f,.68f));
    glm::vec3 eye=state.position, direction=cameraDirection(state);
    r.prepareTrees(eye);
    // 阴影范围跟随视点；远眺时扩大覆盖，贴近地面时恢复更多细节。
    float distanceToGround = direction.y < -.05f ? glm::clamp((20.0f-eye.y)/direction.y,0.0f,1800.0f) : 80.0f;
    glm::vec3 focus = eye + direction * distanceToGround;
    focus.x=glm::clamp(focus.x,r.heights.x0,r.heights.x1);
    focus.z=glm::clamp(focus.z,r.heights.z0,r.heights.z1);
    focus.y=r.heights.at(focus.x,focus.z);
    float shadowRadius=glm::clamp(120.0f+std::max(0.0f,eye.y-focus.y)*.55f,140.0f,950.0f);
    r.lightSpace = glm::ortho(-shadowRadius,shadowRadius,-shadowRadius,shadowRadius,1.0f,3000.0f)
        * glm::lookAt(focus+sun*1200.0f,focus,glm::vec3(0,1,0));
    if (state.shadows) {
        r.depthPass = true;
        glViewport(0,0,SHADOW_SIZE,SHADOW_SIZE); glBindFramebuffer(GL_FRAMEBUFFER,r.shadowFbo);
        glClear(GL_DEPTH_BUFFER_BIT); glEnable(GL_DEPTH_TEST); glUseProgram(r.depth);
        glEnable(GL_POLYGON_OFFSET_FILL); glPolygonOffset(2.0f,4.0f);
        glUniformMatrix4fv(glGetUniformLocation(r.depth,"lightSpace"),1,GL_FALSE,glm::value_ptr(r.lightSpace));
        world(r,state);
        glDisable(GL_POLYGON_OFFSET_FILL);
    }

    r.depthPass = false; glBindFramebuffer(GL_FRAMEBUFFER,0); glViewport(0,0,width,height);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST); glUseProgram(r.sky); glBindVertexArray(r.emptyVao); glDrawArrays(GL_TRIANGLES,0,3);
    glEnable(GL_DEPTH_TEST); glUseProgram(r.scene);
    glm::mat4 view = glm::lookAt(eye,eye+direction,glm::vec3(0,1,0));
    glm::mat4 projection = glm::perspective(glm::radians(52.0f),float(width)/float(std::max(height,1)),.4f,4000.0f);
    glUniformMatrix4fv(glGetUniformLocation(r.scene,"view"),1,GL_FALSE,glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(r.scene,"projection"),1,GL_FALSE,glm::value_ptr(projection));
    glUniformMatrix4fv(glGetUniformLocation(r.scene,"lightSpace"),1,GL_FALSE,glm::value_ptr(r.lightSpace));
    glUniform3fv(glGetUniformLocation(r.scene,"lightDirection"),1,glm::value_ptr(sun));
    glUniform3fv(glGetUniformLocation(r.scene,"eyePosition"),1,glm::value_ptr(eye));
    glUniform1f(glGetUniformLocation(r.scene,"timeSeconds"),seconds);
    glUniform1i(glGetUniformLocation(r.scene,"shadowsEnabled"),state.shadows ? 1 : 0);
    glUniform1i(glGetUniformLocation(r.scene,"colorTexture"),0);
    glUniform1i(glGetUniformLocation(r.scene,"shadowMap"),1);
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D,r.shadowTexture);
    world(r,state);
    r.placeLabels.draw(projection*view,eye,width,height,state.placeNames);
}

void saveScreenshot(const std::string& path, int width, int height) {
    std::vector<unsigned char> pixels(static_cast<size_t>(width) * height * 3);
    glPixelStorei(GL_PACK_ALIGNMENT,1); glReadBuffer(GL_BACK);
    glReadPixels(0,0,width,height,GL_RGB,GL_UNSIGNED_BYTE,&pixels[0]);
    std::ofstream file(path.c_str(),std::ios::binary);
    if (!file) throw std::runtime_error("Cannot write screenshot: " + path);
    file << "P6\n" << width << " " << height << "\n255\n";
    for (int y=height-1;y>=0;--y)
        file.write(reinterpret_cast<const char*>(&pixels[static_cast<size_t>(y)*width*3]),width*3);
    if (!file) throw std::runtime_error("Cannot complete screenshot: " + path);
}

int selfTest() {
    // 验证父子矩阵组合和两个独立交互的边界，而不是仅检查函数被调用。
    glm::mat4 parent = moved(glm::mat4(1),3,0,5);
    glm::mat4 building = turned(parent,90,0,1,0);
    glm::mat4 floor = moved(building,0,4,0);
    glm::mat4 window = moved(floor,2,0,1);
    glm::mat4 frame = moved(window,0,.2f,0);
    glm::vec4 point = frame * glm::vec4(0,0,0,1);
    if (glm::length(glm::vec3(point)-glm::vec3(4,4.2f,3)) > .001f) return 1;
    State state;
    ShuttleRoute shuttleRoute;
    updateObjects(state,1.0f,false,true,false,true,shuttleRoute.length());
    if (state.armillaryAngle != 55.0f || state.busDistance != 28.0f) return 2;
    updateObjects(state,100.0f,false,false,false,true,shuttleRoute.length());
    if (std::abs(state.busDistance-shuttleRoute.length())>.01f) return 3;
    State autoShuttle; autoShuttle.busAuto=true;
    updateObjects(autoShuttle,1.0f,false,false,false,false,shuttleRoute.length());
    if (std::abs(autoShuttle.busDistance-28.0f)>.01f) return 18;
    autoShuttle.busDistance=shuttleRoute.length()-1.0f;
    updateObjects(autoShuttle,1.0f,false,false,false,false,shuttleRoute.length());
    if (autoShuttle.busAuto || std::abs(autoShuttle.busDistance-shuttleRoute.length())>.01f) return 19;
    SwanRoute swanRoute;
    SwanPose swanStart=swanRoute.pose(0,0), swanLater=swanRoute.pose(10,0);
    if (glm::length(swanLater.position-swanStart.position)<2.0f ||
        std::abs(swanStart.position.y-swanLater.position.y)>.001f ||
        std::abs(glm::length(swanStart.tangent)-1.0f)>.001f) return 20;
    State swanAnimation;
    updateObjects(swanAnimation,1.0f,false,false,false,false,shuttleRoute.length());
    swanAnimation.swansActive=false;
    updateObjects(swanAnimation,2.0f,false,false,false,false,shuttleRoute.length());
    if (std::abs(swanAnimation.swanTime-1.0f)>.001f) return 21;
    KaifengEntry kaifengEntry;
    glm::vec4 closed=kaifengEntry.leaf(glm::mat4(1),false,0)*glm::vec4(2.0f,1.0f,0,1);
    glm::vec4 opened=kaifengEntry.leaf(glm::mat4(1),false,1)*glm::vec4(2.0f,1.0f,0,1);
    if (glm::length(glm::vec3(opened-closed))<1.5f) return 22;
    State doorAnimation; doorAnimation.kaifengDoorOpen=true;
    updateObjects(doorAnimation,1.0f,false,false,false,false,shuttleRoute.length());
    if (std::abs(doorAnimation.kaifengDoor-1.0f)>.001f) return 23;
    doorAnimation.kaifengDoorOpen=false;
    updateObjects(doorAnimation,1.0f,false,false,false,false,shuttleRoute.length());
    if (std::abs(doorAnimation.kaifengDoor)>.001f) return 24;
    for (int i=0;i<TREE_BARK;++i) {
        Image img = readPPM(std::string("assets/")+textureFiles[i]+".ppm");
        if (img.width < 16 || img.height < 16) return 4;
    }
    if (readMesh("assets/tree-bark.mesh").empty() || readMesh("assets/tree-leaves.mesh").empty() || readMesh("assets/palm.mesh").empty()) return 5;
    TerrainHeights terrain;
    if (terrain.at(-126,-103) < -50 || terrain.at(-126,-103) > 150) return 6;
    // 该位置位于北夏馆局部细分地形内，防止运行时误用下方的粗网格。
    if (terrain.at(141.8f,-524.1f)<18.0f || terrain.at(141.8f,-524.1f)>20.0f) return 26;
    if (terrain.at(-302.0f,-39.1f)<24.0f || terrain.at(-302.0f,-39.1f)>25.8f) return 27;
    if (terrain.at(260.0f,-43.0f)<12.6f || terrain.at(260.0f,-43.0f)>13.3f) return 28;
    if (terrain.at(501.0f,-506.0f)<15.9f || terrain.at(501.0f,-506.0f)>16.8f) return 29;
    if (terrain.at(-264.0f,-609.0f)<25.4f || terrain.at(-264.0f,-609.0f)>26.0f) return 30;
    if (terrain.at(-287.9f,-320.6f)<29.5f || terrain.at(-287.9f,-320.6f)>30.1f) return 31;
    if (terrain.at(195.1f,-426.6f)<12.0f || terrain.at(195.1f,-426.6f)>12.5f) return 32;
    if (terrain.at(-379.0f,-94.0f)<24.5f || terrain.at(-379.0f,-94.0f)>25.3f) return 33;
    const std::vector<GateView> gateViews=readGateViews();
    if (gateViews.size()!=11 || readModelLayerSpecs("assets/campus-gates.tsv").size()<40) return 34;
    for (const auto& layer:readModelLayerSpecs("assets/campus-south-spine.tsv"))
        if (readMesh("assets/"+layer.mesh).empty() || readPPM("assets/"+layer.texture).width<16) return 38;
    for (const GateView& gate:gateViews) {
        if (glm::length(gate.target-gate.eye)<1 || gate.eye.y<terrain.at(gate.eye.x,gate.eye.z)+1.7f) return 35;
    }
    for (const char* index:{"assets/campus-qiaoyuan-dorms.tsv","assets/campus-south-dorms.tsv"}) {
        const std::vector<ModelLayerSpec> layers=readModelLayerSpecs(index);
        std::vector<std::string> ids;
        for (const ModelLayerSpec& layer:layers) {
            if (std::find(ids.begin(),ids.end(),layer.id)==ids.end()) ids.push_back(layer.id);
            if (readMesh("assets/"+layer.mesh).empty()) return 38;
            if (layer.texture!="-" && readPPM("assets/"+layer.texture).width<16) return 38;
        }
        if (ids.size()!=(std::string(index).find("qiaoyuan")!=std::string::npos ? 5u : 6u)) return 38;
    }
    const std::vector<PlaceRecord> places=readPlaces();
    if (places.size()<80) return 36;
    for (const char* id:{"dujuan","haibin","qiaoxiang","zhiteng","shop-seven","gate-liye","gate-zhiyuan"}) {
        bool found=false;for (const PlaceRecord& place:places) if (place.id==id) found=true;
        if (!found) return 37;
    }
    for (int i=0;i<=8;++i) {
        ShuttlePose pose=shuttleRoute.pose(shuttleRoute.length()*i/8.0f,terrain);
        if (!std::isfinite(pose.position.x) || !std::isfinite(pose.position.y) || !std::isfinite(pose.position.z) ||
            !std::isfinite(pose.yaw) || !std::isfinite(pose.pitch) || !std::isfinite(pose.roll) ||
            std::abs(pose.position.y-terrain.at(pose.position.x,pose.position.z)-.035f)>.5f ||
            std::abs(glm::length(pose.tangent)-1.0f)>.01f ||
            std::abs(pose.pitch)>45.0f || std::abs(pose.roll)>45.0f) return 17;
    }
    WalkSurfaces walking;
    float crossingGround=terrain.at(409.4f,138.0f);
    float walkingGround=walking.at(409.4f,138.0f,crossingGround,crossingGround);
    float walkingLower=walking.at(409.4f,138.0f,13.0f,crossingGround);
    float walkingUpper=walking.at(409.4f,138.0f,17.0f,crossingGround);
    if (std::abs(walkingGround-crossingGround)>.05f || walkingLower-crossingGround<8.0f ||
        walkingUpper-walkingLower<3.0f) return 14;
    // 从下层桥板横向进入平台，再走到斋区步道，双向检查相机能拾取每一级。
    const glm::vec2 portalRoute[]={{302.164f,-63.200f},{296.156f,-58.744f},
                                   {288.698f,-56.709f},{282.978f,-49.626f}};
    for (bool reverse:{false,true}) {
        float foot = reverse ? 12.232f : 11.182f;
        for (int leg=0;leg<3;++leg) {
            const glm::vec2 a=portalRoute[reverse ? 3-leg : leg];
            const glm::vec2 b=portalRoute[reverse ? 2-leg : leg+1];
            const int samples=static_cast<int>(std::ceil(glm::length(b-a)*10));
            for (int i=0;i<=samples;++i) {
                const glm::vec2 p=glm::mix(a,b,static_cast<float>(i)/samples);
                const float next=walking.at(p.x,p.y,foot,terrain.at(p.x,p.y));
                if (std::abs(next-foot)>.33f) return 40;
                foot=next;
            }
        }
        if (std::abs(foot-(reverse ? 11.182f : 12.232f))>.07f) return 40;
    }
    const float spineGround=terrain.at(351.0f,413.3f);
    const float spineLower=walking.at(351.0f,413.3f,spineGround,spineGround);
    const float spineUpper=walking.at(351.0f,413.3f,7.15f,spineGround);
    if (std::abs(spineLower-spineGround-.06f)>.01f || spineUpper-spineLower<3.7f) return 39;
    float slopeLow=terrain.at(-149.7f,167.5f), slopeHigh=terrain.at(-151.036f,169.5f);
    if (slopeHigh-slopeLow<.3f ||
        std::abs(walking.at(-151.036f,169.5f,slopeLow,slopeHigh)-slopeHigh)>.05f ||
        std::abs(walking.at(-149.7f,167.5f,slopeHigh,slopeLow)-slopeLow)>.05f) return 15;
    // 贴地步行应拾取新增三维台阶的踏面，而不是停留在下方的地形上。
    float stairGround=terrain.at(-246.98f,-7.0f);
    float stairLevel=walking.at(-246.98f,-7.0f,stairGround,stairGround);
    if (stairLevel-stairGround<.035f || stairLevel-stairGround>.55f) return 16;
    float kaifengGround=terrain.at(-574.58f,501.497f);
    float kaifengStep=walking.at(-574.58f,501.497f,kaifengGround,kaifengGround);
    if (kaifengStep-kaifengGround<.30f || kaifengStep-kaifengGround>.80f) return 25;
    resetView(state,2);
    if (glm::length(cameraDirection(state)) < .99f || state.position.y < terrain.at(state.position.x,state.position.z)) return 7;
    if (readMesh("assets/campus-terrain.mesh").empty() || readMesh("assets/campus-water.mesh").empty()) return 8;
    State moving; moving.yaw=-90; moving.position=glm::vec3(0,10,0);
    moveCamera(moving,5,true,false,false,false,false,true);
    if (glm::length(moving.position-glm::vec3(0,15,-5))>.001f) return 9;
    const char* detailedMeshes[] = {"campus-north-city-ground.mesh", "campus-water-bank.mesh", "campus-bridge.mesh", "campus-bridge-roof.mesh",
        "campus-bridge-support.mesh", "campus-stadium-canopy.mesh", "campus-archive-seal.mesh",
        "campus-stadium-surface.mesh", "campus-stadium-goals.mesh",
        "campus-stands-orange.mesh", "campus-stands-yellow.mesh", "campus-stands-base.mesh", "campus-stadium-lamps.mesh",
        "campus-wall-teng.mesh", "campus-wall-perforated.mesh", "campus-zhai-stone.mesh", "campus-zhai-paving.mesh", "campus-basketball.mesh", "campus-tennis.mesh",
        "campus-sports-netting.mesh", "campus-basketball-boards.mesh", "campus-basketball-rims.mesh",
        "campus-wall-fortress.mesh", "campus-wall-activity.mesh", "campus-activity-foundation.mesh", "campus-teachers-foundation.mesh", "campus-northsports-foundation.mesh", "campus-southdorm-foundation.mesh", "campus-windzhai-foundation.mesh", "campus-huiwenroad-foundation.mesh", "campus-facultyroad-foundation.mesh", "campus-stone-plaza.mesh", "campus-south-pool.mesh", "campus-stone-canopies.mesh",
        "campus-sundial-face.mesh", "campus-sundial-stone.mesh",
        "campus-alumni-garden.mesh", "campus-wall-huiwen.mesh", "campus-huiwen-detail.mesh",
        "campus-huiwen-terrace.mesh", "campus-huiwen-red.mesh", "campus-huiwen-steps.mesh",
        "campus-zhounan-lower.mesh", "campus-zhounan-truss.mesh", "campus-zhaonan-glass.mesh",
        "campus-zhounan-yellow.mesh", "campus-zhounan-pots.mesh", "campus-zhounan-plants.mesh", "campus-zhounan-ceiling.mesh",
        "campus-zhounan-zhai-treads.mesh", "campus-zhounan-zhai-walls.mesh",
        "campus-huixing-steps.mesh", "campus-huixing-hedges.mesh", "campus-wall-huiyan.mesh",
        "campus-zhiyi-shell.mesh", "campus-zhiyi-skywalk.mesh", "campus-zhiyi-skywalk-rails.mesh",
        "campus-zhizhen-shell.mesh", "campus-zhizhen-glass.mesh", "campus-zhizhen-side.mesh",
        "campus-zhizhen-frame.mesh", "campus-zhizhen-podium.mesh", "campus-studio-shell.mesh",
        "campus-zhizhi-shell.mesh", "campus-zhizhi-glass.mesh", "campus-zhizhi-frame.mesh",
        "campus-l4-glass.mesh", "campus-l4-frame.mesh", "campus-l4-garden.mesh", "campus-l4-equipment.mesh",
        "campus-zhigong-shell.mesh", "campus-zhigong-glass.mesh", "campus-zhigong-dark.mesh",
        "campus-zhigong-perforated.mesh", "campus-zhigong-louvers.mesh", "campus-zhigong-rails.mesh",
        "campus-zhigong-steps.mesh", "campus-zhigong-sign.mesh", "campus-road-surface.mesh", "campus-road-markings.mesh", "campus-road-curbs.mesh", "campus-dujuan-trail.mesh", "campus-dujuan-boardwalk.mesh", "campus-canteen-glass.mesh", "campus-canteen-frame.mesh",
        "campus-network-sidewalk.mesh", "campus-network-paths.mesh", "campus-network-shrubs.mesh", "campus-network-stairs.mesh",
        "campus-lide-walk.mesh", "campus-lide-walk-edge.mesh",
        "campus-lide-corridor-paving.mesh", "campus-lide-corridor-lawn.mesh", "campus-lide-corridor-bed.mesh",
        "campus-lide-corridor-curbs.mesh", "campus-lide-corridor-leaves.mesh",
        "campus-network-verges.mesh", "campus-network-verge-grass.mesh", "campus-network-lamps.mesh", "campus-network-lamp-heads.mesh",
        "campus-wenshan-ancient-stone.mesh", "campus-wenshan-ancient-stone-planting.mesh",
        "campus-wenshan-swan-adult.mesh", "campus-wenshan-swan-feather.mesh", "campus-wenshan-swan-bill.mesh",
        "campus-wenshan-swan-bill-tip.mesh", "campus-wenshan-swan-adult-eye.mesh", "campus-wenshan-swan-chick.mesh", "campus-wenshan-swan-chick-wing.mesh",
        "campus-wenshan-swan-chick-bill.mesh", "campus-wenshan-swan-chick-eye.mesh", "campus-wenshan-swan-wake.mesh",
        "campus-kaifeng-shell.mesh", "campus-kaifeng-roof.mesh", "campus-kaifeng-windows.mesh", "campus-kaifeng-yellow.mesh", "campus-kaifeng-red.mesh",
        "campus-kaifeng-frame.mesh", "campus-kaifeng-rail.mesh", "campus-kaifeng-stairs.mesh", "campus-kaifeng-apron.mesh", "campus-kaifeng-interior.mesh",
        "campus-kaifeng-furniture.mesh", "campus-kaifeng-door-glass.mesh", "campus-kaifeng-door-frame.mesh",
        "campus-shuttle-body.mesh", "campus-shuttle-metal.mesh", "campus-shuttle-interior.mesh", "campus-shuttle-seats.mesh",
        "campus-shuttle-glass.mesh", "campus-shuttle-lamps.mesh", "campus-shuttle-sign.mesh",
        "campus-wall-xiyuan.mesh", "campus-xiyuan-frame.mesh", "campus-xiyuan-railing.mesh",
        "campus-qiaoyuan-garden-walk.mesh", "campus-qiaoyuan-garden-brick.mesh", "campus-qiaoyuan-garden-frame.mesh", "campus-qiaoyuan-garden-words.mesh",
        "campus-xiyuan-shops-glass.mesh", "campus-xiyuan-shops-signs.mesh", "campus-xiyuan-shops-paving.mesh",
        "campus-xiyuan-shops-membrane.mesh", "campus-xiyuan-shops-awning-frame.mesh", "campus-xiyuan-shops-manner-wall.mesh",
        "campus-xiyuan-shops-opposite-paving.mesh", "campus-xiyuan-shops-plant-bed.mesh", "campus-xiyuan-shops-plant-leaves.mesh",
        "campus-canteen-roof.mesh", "campus-canteen-sign.mesh", "campus-zhigong-cars-light.mesh",
        "campus-zhigong-cars-dark.mesh", "campus-zhigong-cars-glass.mesh", "campus-zhigong-cars-wheels.mesh", "campus-zhigong-chargers.mesh",
        "campus-studio-sign.mesh", "campus-zhili-glass.mesh", "campus-zhili-frame.mesh", "campus-zhili-side.mesh",
        "campus-zhili-a-shell.mesh", "campus-zhili-a-glass.mesh", "campus-zhili-a-frame.mesh", "campus-zhili-a-sign.mesh",
        "campus-zhili-high-walkways.mesh", "campus-zhili-high-walkway-rails.mesh",
        "campus-zhili-crown.mesh", "campus-zhili-crown-glass.mesh", "campus-zhili-crown-frame.mesh",
        "campus-huicui-shell.mesh", "campus-huicui-glass.mesh", "campus-huicui-upper.mesh",
        "campus-huicui-stage.mesh", "campus-huicui-frames.mesh", "campus-huicui-skylights.mesh",
        "campus-huizi-shell.mesh", "campus-huizi-ground.mesh", "campus-huizi-canopy.mesh", "campus-huizi-paving.mesh",
        "campus-huizi-roof.mesh", "campus-huizi-equipment.mesh",
        "campus-huiyuan-shell.mesh", "campus-huiyuan-ground.mesh", "campus-huiyuan-sign.mesh",
        "campus-huiyuan-motto.mesh", "campus-huiyuan-paving.mesh", "campus-huiyuan-planting.mesh", "campus-huiyuan-entry-roof.mesh",
        "campus-yanhui-roof.mesh", "campus-yanhui-under.mesh", "campus-yanhui-steel.mesh",
        "campus-yanhui-stone.mesh", "campus-yanhui-floor.mesh", "campus-yanhui-seats.mesh", "campus-yanhui-seat-detail.mesh",
        "campus-yanhui-stage.mesh", "campus-yanhui-screen.mesh", "campus-yanhui-steps.mesh", "campus-yanhui-seams.mesh",
        "campus-yanhui-pine-trunks.mesh", "campus-yanhui-pine-crowns.mesh",
        "campus-time-paving.mesh", "campus-time-curb.mesh", "campus-time-hedges.mesh", "campus-time-hedge-leaves.mesh",
        "campus-yuanping-shell.mesh", "campus-yuanping-base.mesh", "campus-yuanping-foundation.mesh", "campus-yuanping-roof.mesh",
        "campus-yuanping-truss.mesh", "campus-yuanping-relief.mesh", "campus-yuanping-seal.mesh",
        "campus-yuanping-glass.mesh", "campus-yuanping-steps.mesh", "campus-yuanping-name.mesh", "campus-yuanping-garden.mesh", "campus-beixia-shell.mesh",
        "campus-beixia-roof.mesh", "campus-beixia-frame.mesh", "campus-beixia-seats.mesh",
        "campus-beixia-deck.mesh", "campus-beixia-lanes.mesh", "campus-beixia-practice.mesh", "campus-beixia-coping.mesh", "campus-beixia-under.mesh",
        "campus-golf-floor.mesh", "campus-golf-roof.mesh", "campus-golf-frame.mesh", "campus-golf-mats.mesh", "campus-golf-signs.mesh"};
    for (size_t i=0;i<sizeof(detailedMeshes)/sizeof(detailedMeshes[0]);++i)
        if (readMesh(std::string("assets/")+detailedMeshes[i]).empty()) return 10;
    readMesh("assets/campus-annotated-shrubs.mesh",true);
    if (readPPM("assets/zhounan-zhai-tile.ppm").width!=256) return 40;
    if (readPPM("assets/studio-sign.ppm").width < 16 || readPPM("assets/gate-address.ppm").width < 16 ||
        readPPM("assets/yuanping-relief.ppm").width < 16 || readPPM("assets/yuanping-name.ppm").width < 16 ||
        readPPM("assets/huizi-roof.ppm").width < 16 || readPPM("assets/huizi-equipment.ppm").width < 16 ||
        readPPM("assets/facade-huiyan.ppm").width < 16 || readPPM("assets/stadium-seat.ppm").width < 16 ||
        readPPM("assets/stadium-surface.ppm").width < 16 ||
        readPPM("assets/zhiyuan-sign-base.ppm").width < 16) return 10;
    std::ifstream words("assets/zhiyuan-sign-words.rgba",std::ios::binary|std::ios::ate);
    if (!words || words.tellg() != std::streamoff(8+1024*256*4)) return 10;
    for (int view=7;view<=ZHAI_ROADSIDE_STAIR_VIEW;++view) {
        resetView(state,view);
        if (state.position.y < terrain.at(state.position.x,state.position.z) || glm::length(cameraDirection(state)) < .99f) return 11;
    }
    // 中央塔框架应从主体屋顶延伸到设计院给出的 95 米高度。
    std::vector<Vertex> towerFrame=huixingFrameGeometry();
    float low=1000.0f,high=-1000.0f;
    for (size_t i=0;i<towerFrame.size();++i) {
        low=std::min(low,towerFrame[i].p.y);
        high=std::max(high,towerFrame[i].p.y);
    }
    if (towerFrame.size()<1000 || std::abs(low-72.5f)>.1f || std::abs(high-95.0f)>.1f) return 12;
    if (readPPM("assets/kaifeng-tiles.ppm").width != 256 || readPPM("assets/wenshan-swan-feathers.ppm").width != 512 || readPPM("assets/wenshan-ancient-stone.ppm").width != 2048 || readPPM("assets/shuttle-sign.ppm").width != 512 || readPPM("assets/campus-sidewalk.ppm").width != 256 || readPPM("assets/xiyuan-shop-signs.ppm").width != 1024 || readPPM("assets/facade-xiyuan.ppm").width != 256 ||
        readPPM("assets/campus-asphalt.ppm").width != 256 || readPPM("assets/dujuan-boardwalk.ppm").width != 256 || readPPM("assets/canteen-sign.ppm").width != 512 ||
        readPPM("assets/facade-huixing.ppm").width != 512 ||
        readPPM("assets/facade-huixing-core.ppm").width != 256) return 13;
    // 可选独立资产也纳入课程程序自检；缺包仍使用旧门，损坏包必须失败。
    std::unique_ptr<CampusAsset> independentGate=CampusAsset::loadOptional("assets/modeling/lide-gate");
    if (independentGate) independentGate->report(0.0f);
    std::cout << "Self-test passed: hierarchy, object controls, campus data, free camera.\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    GLFWwindow* window = NULL;
    bool silentErrors = false;
    try {
        enterExecutableDirectory();
        std::string capture;
        bool self = false, benchmark = false, noShadows = false, captureVisible = false,noLabels=false;
        bool assetCheck=false,hasAssetAngle=false,yuanpingWalkCheck=false,stadiumWalkCheck=false,hasCamera=false;
        bool northEntryCheck=false,northUpperCheck=false;
        glm::vec3 cameraEye,cameraTarget;
        float assetAngle=0.0f;
        std::string gatePreset;
        std::string corridorPreset;
        double captureDelay = 0.0;
        int viewPreset = 1;
        for (int i=1;i<argc;++i) {
            std::string arg(argv[i]);
            if (arg == "--self-test") self = true;
            else if (arg == "--benchmark") benchmark = true;
            else if (arg == "--no-shadows") noShadows = true;
            else if (arg == "--no-labels") noLabels = true;
            else if (arg == "--capture-visible") captureVisible = true;
            else if (arg == "--asset-check") assetCheck = true;
            else if (arg == "--yuanping-walk-check") yuanpingWalkCheck = true;
            else if (arg == "--stadium-walk-check") stadiumWalkCheck = true;
            else if (arg == "--north-library-walk-check") northEntryCheck = true;
            else if (arg == "--north-library-upper-check") northUpperCheck = true;
            else if (arg.find("--camera=") == 0) {
                std::string values=arg.substr(9); std::replace(values.begin(),values.end(),',',' ');
                std::istringstream row(values); std::string extra;
                if (!(row>>cameraEye.x>>cameraEye.y>>cameraEye.z>>cameraTarget.x>>cameraTarget.y>>cameraTarget.z) || (row>>extra))
                    throw std::runtime_error("--camera expects eyeX,eyeY,eyeZ,targetX,targetY,targetZ");
                for (int k=0;k<3;++k) if (!std::isfinite(cameraEye[k]) || !std::isfinite(cameraTarget[k]))
                    throw std::runtime_error("Non-finite camera");
                if (glm::length(cameraTarget-cameraEye)<.01f) throw std::runtime_error("Empty camera direction");
                hasCamera=true;
            }
            else if (arg.find("--asset-angle=") == 0) {
                const std::string value=arg.substr(14); char* end=NULL;
                assetAngle=std::strtof(value.c_str(),&end);
                if (value.empty() || !end || *end || !std::isfinite(assetAngle))
                    throw std::runtime_error("Invalid --asset-angle: "+value);
                hasAssetAngle=true;
            }
            else if (arg.find("--capture=") == 0) capture = arg.substr(10);
            else if (arg.find("--capture-delay=") == 0) captureDelay = std::max(0.0, std::atof(arg.substr(16).c_str()));
            else if (arg.find("--view=") == 0) viewPreset = std::atoi(arg.substr(7).c_str());
            else if (arg.find("--gate=") == 0) gatePreset = arg.substr(7);
            else if (arg.find("--corridor=") == 0) corridorPreset = arg.substr(11);
            else throw std::runtime_error("Unknown argument: " + arg);
        }
        silentErrors = self || benchmark || assetCheck || yuanpingWalkCheck || stadiumWalkCheck || northEntryCheck || northUpperCheck || !capture.empty();
        if (yuanpingWalkCheck) return verifyEntryWalk("assets/campus-yuanping-entry-routes.tsv",5,false);
        if (stadiumWalkCheck) return verifyEntryWalk("assets/campus-stadium-entry-routes.tsv",3,true);
        if (northEntryCheck) return verifyEntryWalk("assets/campus-north-library-entry-routes.tsv",3,2);
        if (northUpperCheck) return verifyEntryWalk("assets/campus-north-library-upper-routes.tsv",4,0);
        if (assetCheck) {
            std::unique_ptr<CampusAsset> asset=CampusAsset::loadOptional("assets/modeling/lide-gate");
            if (!asset) {
                if (hasAssetAngle) throw std::runtime_error("--asset-angle requires the Lide asset package");
                std::cout << "{\"status\":\"absent\",\"present\":false}\n";
            } else {
                asset->validateAngle(assetAngle);
                std::cout << asset->report(assetAngle);
            }
            return 0;
        }
        if (self) return selfTest();
        if (!glfwInit()) throw std::runtime_error("GLFW initialization failed");
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3); glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
        glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE); glfwWindowHint(GLFW_SAMPLES,4);
        if ((!capture.empty() && !captureVisible) || benchmark) glfwWindowHint(GLFW_VISIBLE,GLFW_FALSE);
        window = glfwCreateWindow(1280,800,u8"2024270223_王锦政_期末大作业",NULL,NULL);
        if (!window) throw std::runtime_error("OpenGL window creation failed");
        glfwMakeContextCurrent(window); glfwSwapInterval(benchmark ? 0 : 1);
        if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)))
            throw std::runtime_error("GLAD initialization failed");
        std::cout << "OpenGL renderer: " << glGetString(GL_RENDERER) << "\n";
        printHelp();
        State state; resetView(state,viewPreset); if (noShadows) state.shadows = false;
        if (noLabels) state.placeNames=false;
        if (!gatePreset.empty()) {
            const std::vector<GateView>& views=campusGateViews();
            size_t index=0;while (index<views.size() && views[index].id!=gatePreset) ++index;
            if (index==views.size()) throw std::runtime_error("Unknown campus gate: "+gatePreset);
            selectGateView(state,static_cast<int>(index));
        }
        if (!corridorPreset.empty()) {
            // 样板机位单独保存，保持校门巡览的顺序和数量。
            const std::vector<GateView> views=readGateViews("assets/campus-lide-corridor-views.tsv");
            auto preset=std::find_if(views.begin(),views.end(),[&](const GateView& view) {
                return view.id==corridorPreset;
            });
            if (preset==views.end()) throw std::runtime_error("Unknown Lide corridor view: "+corridorPreset);
            aimCamera(state,preset->eye,preset->target);
            state.speed=10;
        }
        glfwSetWindowUserPointer(window,&state); glfwSetKeyCallback(window,keyCallback);
        if (hasCamera) { aimCamera(state,cameraEye,cameraTarget); state.groundFollow=false; }
        glfwSetMouseButtonCallback(window,mouseButton); glfwSetCursorPosCallback(window,cursorPosition);
        glfwSetScrollCallback(window,scrollCallback);
        glEnable(GL_DEPTH_TEST); glEnable(GL_MULTISAMPLE);
        {
            Renderer renderer;
            if (renderer.lideAsset && renderer.lideAsset->hasInteraction()) {
                state.lideAvailable=true; state.lideMaxAngle=renderer.lideAsset->maximumAngle();
            }
            if (hasAssetAngle) {
                if (!renderer.lideAsset) throw std::runtime_error("--asset-angle requires the Lide asset package");
                renderer.lideAsset->validateAngle(assetAngle);
                state.lideAngle=assetAngle; state.lideLowered=std::abs(assetAngle)>1e-6f;
            }
            // 加载完成后刷新控制台，便于确认交互已经可用。
            std::cout << "Campus scene ready.\n" << std::flush;
            double previous = glfwGetTime(), start = previous;
            int frames = 0;
            while (!glfwWindowShouldClose(window)) {
                double now = glfwGetTime();
                float dt = static_cast<float>(std::min(now - previous,.05)); previous = now;
                if (!benchmark) {
                    updateObjects(state,dt,glfwGetKey(window,GLFW_KEY_LEFT)==GLFW_PRESS,glfwGetKey(window,GLFW_KEY_RIGHT)==GLFW_PRESS,
                                  glfwGetKey(window,GLFW_KEY_J)==GLFW_PRESS,glfwGetKey(window,GLFW_KEY_L)==GLFW_PRESS,
                                  renderer.shuttleRoute.length());
                    bool fast=glfwGetKey(window,GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS;
                    float speed = dt * state.speed * (fast ? 2.5f : 1.0f);
                    if (state.groundFollow) speed=std::min(speed,dt*(fast ? 12.0f : 6.0f));
                    float previousFoot=state.position.y-1.8f;
                    moveCamera(state,speed,
                        glfwGetKey(window,GLFW_KEY_W)==GLFW_PRESS,glfwGetKey(window,GLFW_KEY_A)==GLFW_PRESS,
                        glfwGetKey(window,GLFW_KEY_S)==GLFW_PRESS,glfwGetKey(window,GLFW_KEY_D)==GLFW_PRESS,
                        glfwGetKey(window,GLFW_KEY_Q)==GLFW_PRESS,glfwGetKey(window,GLFW_KEY_E)==GLFW_PRESS);
                    state.position.x=glm::clamp(state.position.x,renderer.heights.x0+1,renderer.heights.x1-1);
                    state.position.z=glm::clamp(state.position.z,renderer.heights.z0+1,renderer.heights.z1-1);
                    float ground=renderer.heights.at(state.position.x,state.position.z);
                    if (state.groundFollow)
                        state.position.y=renderer.walking.at(state.position.x,state.position.z,previousFoot,ground)+1.8f;
                    else state.position.y=std::max(state.position.y,ground+1.8f);
                    if (state.busFollow) {
                        ShuttlePose pose=renderer.shuttleRoute.pose(state.busDistance,renderer.heights);
                        glm::vec3 forward(pose.tangent.x,0,pose.tangent.y);
                        glm::vec3 eye=pose.position-forward*8.0f+glm::vec3(0,4.0f,0);
                        eye.y=std::max(eye.y,renderer.heights.at(eye.x,eye.z)+1.8f);
                        aimCamera(state,eye,pose.position+forward*2.0f+glm::vec3(0,1.2f,0));
                    }
                    if (state.swanFollow) {
                        SwanPose pose=renderer.swanRoute.pose(state.swanTime,0);
                        glm::vec3 forward(pose.tangent.x,0,pose.tangent.y), side(-forward.z,0,forward.x);
                        glm::vec3 eye=pose.position-forward*2.5f+side*6.0f+glm::vec3(0,2.2f,0);
                        eye.y=std::max(eye.y,renderer.heights.at(eye.x,eye.z)+1.8f);
                        aimCamera(state,eye,pose.position-forward*1.4f+glm::vec3(0,.4f,0));
                    }
                } else if (benchmark) {
                    state.yaw += dt * 8.0f;
                }
                int width=0,height=0; glfwGetFramebufferSize(window,&width,&height);
                if (width > 0 && height > 0) {
                    render(renderer,state,width,height,static_cast<float>(now));
                    if (!capture.empty() && now - start >= captureDelay) {
                        saveScreenshot(capture,width,height);
                        std::cout << "Capture state: yaw=" << state.yaw << " pitch=" << state.pitch
                                  << " position=" << state.position.x << "," << state.position.y << "," << state.position.z
                                  << " sculpture=" << state.armillaryAngle << " busDistance=" << state.busDistance
                                  << " swanTime=" << state.swanTime << " swansActive=" << state.swansActive << " swanFollow=" << state.swanFollow
                                  << " kaifengDoor=" << state.kaifengDoor
                                  << " placeNames=" << state.placeNames << " visibleLabels=" << renderer.placeLabels.visibleCount
                                  << " gateSelection=" << state.gateSelection
                                  << " groundFollow=" << state.groundFollow
                                  << " shadows=" << state.shadows << "\n";
                        break;
                    }
                    if (benchmark) glFinish();
                    glfwSwapBuffers(window); ++frames;
                }
                glfwPollEvents();
                if (benchmark && frames >= 240) {
                    double elapsed = glfwGetTime()-start;
                    std::cout << "Benchmark: " << frames << " frames / " << elapsed << " s = " << frames/elapsed
                              << " FPS at " << width << "x" << height << " (hidden window, vsync off).\n";
                    break;
                }
            }
        }
        glfwDestroyWindow(window); glfwTerminate();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
#ifdef _WIN32
        if (!silentErrors) MessageBoxA(NULL,e.what(),"Campus scene error",MB_OK | MB_ICONERROR);
#endif
        if (window) glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }
}
