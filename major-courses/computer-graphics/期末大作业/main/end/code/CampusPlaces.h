// 中文名称锚定世界坐标；屏幕避让、距离分级与深度测试共同控制标注。
struct PlaceRecord {
    std::string id,name;
    glm::vec3 anchor;
    glm::vec4 uv;
    glm::vec2 size;
    glm::vec4 bounds;
    float localY;
    float range,priority;
};

std::vector<PlaceRecord> readPlaces() {
    std::ifstream file("assets/campus-places.tsv");
    if (!file) throw std::runtime_error("Cannot read campus places");
    std::vector<PlaceRecord> result;std::string line;
    while (std::getline(file,line)) {
        if (line.empty() || line[0]=='#') continue;
        PlaceRecord p;std::istringstream row(line);
        row>>p.id>>p.anchor.x>>p.anchor.y>>p.anchor.z>>p.uv.x>>p.uv.y>>p.uv.z>>p.uv.w
           >>p.size.x>>p.size.y>>p.range>>p.priority>>p.bounds.x>>p.bounds.y>>p.bounds.z>>p.bounds.w>>p.localY;
        if (!row) throw std::runtime_error("Invalid campus place row");
        std::getline(row,p.name);if (!p.name.empty() && p.name[0]=='\t') p.name.erase(0,1);
        if (p.name.empty() || p.size.x<10 || p.size.y<10 || p.range<1)
            throw std::runtime_error("Invalid campus place metadata");
        result.push_back(p);
    }
    return result;
}

class PlaceLabels {
    struct LabelVertex {glm::vec3 position;glm::vec2 uv,anchorUV;};
    struct Candidate {size_t index;float x,y,z,distance;};
    std::vector<PlaceRecord> places;
    GLuint program,texture,vao,vbo,depthTexture;
    int depthWidth=0,depthHeight=0;
public:
    size_t visibleCount;
    PlaceLabels() : places(readPlaces()),program(0),texture(0),vao(0),vbo(0),depthTexture(0),visibleCount(0) {
        program=loadShaderProgram("shaders/places.vert.glsl","shaders/places.frag.glsl");
        texture=loadRgbaTexture("assets/campus-place-labels.rgba");
        glGenVertexArrays(1,&vao);glGenBuffers(1,&vbo);glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER,vbo);
        glEnableVertexAttribArray(0);glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(LabelVertex),NULL);
        glEnableVertexAttribArray(1);glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,sizeof(LabelVertex),
            reinterpret_cast<void*>(offsetof(LabelVertex,uv)));
        glEnableVertexAttribArray(2);glVertexAttribPointer(2,2,GL_FLOAT,GL_FALSE,sizeof(LabelVertex),
            reinterpret_cast<void*>(offsetof(LabelVertex,anchorUV)));
        glGenTextures(1,&depthTexture);
        glBindVertexArray(0);
    }
    ~PlaceLabels() {
        glDeleteBuffers(1,&vbo);glDeleteVertexArrays(1,&vao);glDeleteTextures(1,&texture);
        glDeleteTextures(1,&depthTexture);glDeleteProgram(program);
    }
    void draw(const glm::mat4& viewProjection,const glm::vec3& eye,int width,int height,bool enabled) {
        visibleCount=0;if (!enabled) return;
        std::vector<Candidate> candidates;
        for (size_t i=0;i<places.size();++i) {
            const PlaceRecord& p=places[i];float distance=glm::length(p.anchor-eye);
            if (distance>p.range) continue;
            glm::vec3 anchor=p.anchor;
            if (distance<250 && eye.y<p.anchor.y-10 && p.bounds.z>p.bounds.x && p.bounds.w>p.bounds.y) {
                // 仰视近处建筑时，把名称放到朝向相机的外缘，避免楼顶文字跑出视野。
                glm::vec2 center((p.bounds.x+p.bounds.z)*.5f,(p.bounds.y+p.bounds.w)*.5f);
                glm::vec2 delta(eye.x-center.x,eye.z-center.y);
                float length=glm::length(delta);
                if (length>1) {
                    float t=std::min((p.bounds.z-p.bounds.x)*.5f/std::max(std::abs(delta.x),.001f),
                                     (p.bounds.w-p.bounds.y)*.5f/std::max(std::abs(delta.y),.001f));
                    glm::vec2 edge=center+delta*(t+2/length);
                    anchor=glm::vec3(edge.x,p.localY,edge.y);
                }
            }
            glm::vec4 clip=viewProjection*glm::vec4(anchor,1);
            if (clip.w<=.4f) continue;
            glm::vec3 ndc=glm::vec3(clip)/clip.w;
            if (std::abs(ndc.x)>.97f || ndc.y<-.95f || ndc.y>.93f || ndc.z< -1 || ndc.z>1) continue;
            Candidate c={i,(ndc.x*.5f+.5f)*width,(ndc.y*.5f+.5f)*height,ndc.z,distance};
            candidates.push_back(c);
        }
        std::stable_sort(candidates.begin(),candidates.end(),[&](const Candidate&a,const Candidate&b) {
            float sa=places[a.index].priority*350+a.distance;
            float sb=places[b.index].priority*350+b.distance;
            return sa<sb;
        });
        float scale=glm::clamp(height/800.0f,.85f,1.8f);
        std::vector<glm::vec4> occupied;std::vector<LabelVertex> vertices;
        for (const Candidate& c:candidates) {
            const PlaceRecord& p=places[c.index];float w=p.size.x*scale,h=p.size.y*scale;
            glm::vec4 box(c.x-w*.5f,c.y,c.x+w*.5f,c.y+h);
            if (box.x<6 || box.z>width-6 || box.w>height-6) continue;
            bool overlap=false;
            for (const glm::vec4& other:occupied)
                if (box.x<other.z+7 && box.z>other.x-7 && box.y<other.w+5 && box.w>other.y-5) {overlap=true;break;}
            if (overlap) continue;
            occupied.push_back(box);
            float x0=box.x/width*2-1,x1=box.z/width*2-1,y0=box.y/height*2-1,y1=box.w/height*2-1;
            glm::vec2 anchorUV(c.x/width,c.y/height);
            LabelVertex corners[]={{{x0,y0,c.z},{p.uv.x,p.uv.y},anchorUV},{{x1,y0,c.z},{p.uv.z,p.uv.y},anchorUV},
                                    {{x1,y1,c.z},{p.uv.z,p.uv.w},anchorUV},{{x0,y1,c.z},{p.uv.x,p.uv.w},anchorUV}};
            const int order[]={0,1,2,0,2,3};for (int index:order) vertices.push_back(corners[index]);
            if (occupied.size()>=20) break;
        }
        visibleCount=occupied.size();if (vertices.empty()) return;
        glUseProgram(program);glUniform1i(glGetUniformLocation(program,"labelAtlas"),0);
        // 只按锚点判断遮挡，整张中文名牌一起隐藏，防止文字被树枝切成半截。
        glActiveTexture(GL_TEXTURE2);glBindTexture(GL_TEXTURE_2D,depthTexture);
        if (depthWidth!=width || depthHeight!=height) {
            glTexImage2D(GL_TEXTURE_2D,0,GL_DEPTH_COMPONENT24,width,height,0,GL_DEPTH_COMPONENT,GL_FLOAT,NULL);
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
            depthWidth=width;depthHeight=height;
        }
        glCopyTexSubImage2D(GL_TEXTURE_2D,0,0,0,0,0,width,height);
        glUniform1i(glGetUniformLocation(program,"sceneDepth"),2);
        glActiveTexture(GL_TEXTURE0);glBindTexture(GL_TEXTURE_2D,texture);
        glBindVertexArray(vao);glBindBuffer(GL_ARRAY_BUFFER,vbo);
        glBufferData(GL_ARRAY_BUFFER,vertices.size()*sizeof(LabelVertex),&vertices[0],GL_DYNAMIC_DRAW);
        glDisable(GL_DEPTH_TEST);glDepthMask(GL_FALSE);glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
        glDrawArrays(GL_TRIANGLES,0,static_cast<GLsizei>(vertices.size()));
        glDisable(GL_BLEND);glDepthMask(GL_TRUE);glEnable(GL_DEPTH_TEST);glBindVertexArray(0);
    }
};
