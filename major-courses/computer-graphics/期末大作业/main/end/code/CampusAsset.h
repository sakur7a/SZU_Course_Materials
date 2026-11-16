#ifndef SZU_CAMPUS_ASSET_H
#define SZU_CAMPUS_ASSET_H

// 局部 SZM1 与节点矩阵直接来自已验证资产包。放置只在校园端应用一次。
class CampusAsset {
    struct Node {
        std::string id,parent;
        int parentIndex;
        glm::mat4 local;
        bool visual;
    };
    struct Material {
        std::string id,texture;
        glm::vec3 color;
        bool doubleSided;
        GLuint gpuTexture;
        Material() : doubleSided(false),gpuTexture(0) {}
    };
    struct Primitive {
        std::string id;
        int node,material;
        std::vector<Vertex> vertices;
        std::unique_ptr<Mesh> mesh;
    };
    struct StaticBatch {
        int material;
        std::unique_ptr<Mesh> mesh;
    };
    struct Anchor {
        std::string id;
        int node;
        glm::vec3 local,expected;
    };
    typedef std::vector<std::vector<std::string> > Rows;
    std::string root;
    std::vector<Node> nodes;
    std::vector<Material> materials;
    std::vector<Primitive> primitives;
    std::vector<StaticBatch> staticBatches;
    std::vector<Anchor> anchors;
    std::map<std::string,int> nodeIds,materialIds;
    glm::mat4 placement;
    int interaction,probeParent,probeDescendant;
    glm::vec3 axis;
    float minAngle,maxAngle,probeAngle;

    static void require(bool ok,const std::string& message) {
        if (!ok) throw std::runtime_error("Lide asset: "+message);
    }
    static std::string path(const std::string& base,const std::string& relative) {
        require(!relative.empty() && relative[0]!='/' && relative[0]!='\\' &&
                relative.find(':')==std::string::npos,"invalid relative path: "+relative);
        std::string part;
        for (size_t i=0;i<=relative.size();++i) {
            if (i==relative.size() || relative[i]=='/' || relative[i]=='\\') {
                require(!part.empty() && part!="." && part!="..","invalid relative path: "+relative);
                part.clear();
            } else part.push_back(relative[i]);
        }
        return base+"/"+relative;
    }
    static Rows readRows(const std::string& file) {
        std::ifstream input(file.c_str(),std::ios::binary);
        require(bool(input),"cannot open "+file);
        Rows result; std::string line;
        while (std::getline(input,line)) {
            if (!line.empty() && line.back()=='\r') line.pop_back();
            if (result.empty() && line.size()>=3 && static_cast<unsigned char>(line[0])==0xef &&
                static_cast<unsigned char>(line[1])==0xbb && static_cast<unsigned char>(line[2])==0xbf)
                line.erase(0,3);
            if (line.empty() || line[0]=='#') continue;
            std::vector<std::string> row; size_t start=0;
            for (;;) {
                const size_t at=line.find('\t',start);
                row.push_back(line.substr(start,at==std::string::npos ? at : at-start));
                if (at==std::string::npos) break;
                start=at+1;
            }
            result.push_back(row);
        }
        require(!input.bad(),"cannot read "+file);
        return result;
    }
    static float number(const std::string& value,const std::string& where) {
        char* end=NULL; const float result=std::strtof(value.c_str(),&end);
        require(!value.empty() && end!=value.c_str() && end && !*end && std::isfinite(result),
                "invalid number in "+where+": "+value);
        return result;
    }
    static glm::vec3 vector3(const std::vector<std::string>& row,size_t offset,const std::string& where) {
        return glm::vec3(number(row[offset],where),number(row[offset+1],where),number(row[offset+2],where));
    }
    static glm::mat4 matrix(const std::vector<std::string>& row,size_t offset,const std::string& where) {
        glm::mat4 value(1.0f);
        for (int col=0;col<4;++col) for (int r=0;r<4;++r)
            value[col][r]=number(row[offset+col*4+r],where);
        const float determinant=glm::determinant(glm::mat3(value));
        require(std::abs(value[0][3])<1e-6f && std::abs(value[1][3])<1e-6f &&
                std::abs(value[2][3])<1e-6f && std::abs(value[3][3]-1.0f)<1e-6f &&
                std::isfinite(determinant) && std::abs(determinant)>1e-10f,"invalid affine matrix in "+where);
        return value;
    }
    static int lookup(const std::map<std::string,int>& index,const std::string& id,const std::string& kind) {
        const auto it=index.find(id); require(it!=index.end(),"unknown "+kind+": "+id); return it->second;
    }
    static bool hasRole(const std::string& roles,const std::string& role) {
        std::istringstream input(roles); std::string item;
        while (std::getline(input,item,',')) if (item==role) return true;
        return false;
    }
    static void checkTexture(const std::string& file) {
        std::ifstream input(file.c_str(),std::ios::binary|std::ios::ate);
        require(bool(input),"cannot open texture "+file);
        const std::streamoff size=input.tellg(); input.seekg(0);
        std::uint32_t width=0,height=0;
        input.read(reinterpret_cast<char*>(&width),4); input.read(reinterpret_cast<char*>(&height),4);
        require(bool(input) && width && height && width<=4096 && height<=4096 &&
                size==std::streamoff(8)+std::streamoff(width)*height*4,"invalid RGBA texture "+file);
    }
    std::vector<glm::mat4> worlds(int rotated,const glm::vec3& rotationAxis,float degrees) const {
        std::vector<glm::mat4> result(nodes.size()); std::vector<int> state(nodes.size(),0);
        std::function<void(int)> visit=[&](int i) {
            require(state[i]!=1,"node hierarchy contains a cycle at "+nodes[i].id);
            if (state[i]==2) return;
            state[i]=1; glm::mat4 local=nodes[i].local;
            // 轴心在动件节点原点；增量右乘局部矩阵，子节点自然继承。
            if (i==rotated) local=local*glm::rotate(glm::mat4(1.0f),glm::radians(degrees),rotationAxis);
            if (nodes[i].parentIndex>=0) {
                visit(nodes[i].parentIndex); result[i]=result[nodes[i].parentIndex]*local;
            } else result[i]=local;
            for (int col=0;col<4;++col) for (int r=0;r<4;++r)
                require(std::isfinite(result[i][col][r]),"non-finite world matrix at "+nodes[i].id);
            state[i]=2;
        };
        for (size_t i=0;i<nodes.size();++i) visit(static_cast<int>(i));
        return result;
    }
    bool descendantOf(int node,int ancestor) const {
        for (int p=node;p>=0;p=nodes[p].parentIndex) if (p==ancestor) return true;
        return false;
    }
    bool dynamic(const Primitive& primitive) const {
        return interaction>=0 && descendantOf(primitive.node,interaction);
    }
    static void appendStaticGeometry(std::vector<Vertex>& output,const std::vector<Vertex>& input,
                                     const glm::mat4& world) {
        const float determinant=glm::determinant(glm::mat3(world));
        require(std::isfinite(determinant) && std::abs(determinant)>std::numeric_limits<float>::min(),
                "invalid static node world matrix");
        const glm::mat3 normalMatrix=glm::transpose(glm::inverse(glm::mat3(world)));
        const bool reverse=determinant<0;
        for (size_t i=0;i<input.size();i+=3) {
            for (size_t corner=0;corner<3;++corner) {
                const size_t source=reverse && corner ? 3-corner : corner;
                Vertex value=input[i+source];
                value.p=glm::vec3(world*glm::vec4(value.p,1));
                // 保留法线长度，使非均匀缩放后的插值与原 shader 一致。
                value.n=normalMatrix*value.n;
                require(std::isfinite(value.p.x) && std::isfinite(value.p.y) && std::isfinite(value.p.z) &&
                        std::isfinite(value.n.x) && std::isfinite(value.n.y) && std::isfinite(value.n.z) &&
                        glm::dot(value.n,value.n)>0,"invalid baked static vertex");
                output.push_back(value);
            }
        }
    }
    static float maxDifference(const glm::mat4& a,const glm::mat4& b) {
        float result=0;
        for (int col=0;col<4;++col) for (int r=0;r<4;++r) result=std::max(result,std::abs(a[col][r]-b[col][r]));
        return result;
    }
    static std::string json(const std::string& text) {
        std::ostringstream out; out << '"';
        for (unsigned char c:text) {
            if (c=='"' || c=='\\') out << '\\' << static_cast<char>(c);
            else if (c<0x20) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
            else out << static_cast<char>(c);
        }
        out << '"'; return out.str();
    }
    static void jsonVector(std::ostream& out,const glm::vec3& v) { out << '[' << v.x << ',' << v.y << ',' << v.z << ']'; }
    static void jsonMatrix(std::ostream& out,const glm::mat4& m) {
        out << '['; for (int i=0;i<16;++i) { if (i) out << ','; out << m[i/4][i%4]; } out << ']';
    }

    explicit CampusAsset(const std::string& directory) : root(directory),placement(1.0f),interaction(-1),
        probeParent(-1),probeDescendant(-1),axis(0,1,0),minAngle(0),maxAngle(0),probeAngle(0) {
        Rows rows=readRows(path(root,"placement.tsv"));
        require(rows.size()==1 && rows[0].size()==17 && rows[0][0]=="P","placement.tsv must contain P and 16 values");
        placement=matrix(rows[0],1,"placement.tsv");
        rows=readRows(path(root,"legacy/nodes.tsv"));
        for (const auto& row:rows) {
            require(row.size()==20 && row[0]=="N" && !row[1].empty() && row[1]!="-","invalid nodes.tsv row");
            Node n; n.id=row[1]; n.parent=row[2]; n.parentIndex=-1; n.local=matrix(row,4,"nodes.tsv");
            n.visual=hasRole(row[3],"visual");
            require(nodeIds.insert(std::make_pair(n.id,static_cast<int>(nodes.size()))).second,"duplicate node "+n.id);
            nodes.push_back(n);
        }
        require(!nodes.empty(),"no nodes");
        for (Node& node:nodes) if (node.parent!="-") node.parentIndex=lookup(nodeIds,node.parent,"parent node");
        const std::vector<glm::mat4> base=worlds(-1,axis,0);

        rows=readRows(path(root,"legacy/materials.tsv"));
        for (const auto& row:rows) {
            require(row.size()==10 && row[0]=="T" && !row[1].empty() && row[1]!="-","invalid materials.tsv row");
            require(row[3]=="OPAQUE","only OPAQUE materials are supported by this campus adapter: "+row[1]);
            const float cutoff=number(row[4],"materials.tsv");
            require(cutoff>=0 && cutoff<=1 && (row[5]=="0" || row[5]=="1"),"invalid material flags: "+row[1]);
            Material m; m.id=row[1]; m.texture=row[2]; m.doubleSided=row[5]=="1";
            for (int i=0;i<4;++i) {
                const float value=number(row[6+i],"materials.tsv");
                require(value>=0 && value<=1,"invalid base color: "+m.id);
                // 资产基色为线性 RGB；旧校园 shader 直接在显示编码空间着色。
                // 仅对新资产作显示近似，不改变旧校园材质或 shader。
                if (i<3) m.color[i]=value<=0.0031308f ? 12.92f*value : 1.055f*std::pow(value,1.0f/2.4f)-0.055f;
            }
            if (m.texture!="-") checkTexture(path(root,m.texture));
            require(materialIds.insert(std::make_pair(m.id,static_cast<int>(materials.size()))).second,"duplicate material "+m.id);
            materials.push_back(m);
        }
        require(!materials.empty(),"no materials");

        std::map<std::string,int> ids;
        rows=readRows(path(root,"legacy/primitives.tsv")); bool visualGeometry=false;
        for (const auto& row:rows) {
            require(row.size()==5 && row[0]=="P" && !row[1].empty(),"invalid primitives.tsv row");
            require(ids.insert(std::make_pair(row[1],1)).second,"duplicate primitive "+row[1]);
            Primitive p; p.id=row[1]; p.node=lookup(nodeIds,row[2],"primitive node");
            p.material=lookup(materialIds,row[3],"primitive material");
            const std::string file=path(root,row[4]);
            std::ifstream input(file.c_str(),std::ios::binary|std::ios::ate);
            require(bool(input),"cannot open mesh "+file); const std::streamoff size=input.tellg();
            p.vertices=readMesh(file);
            require(p.vertices.size()%3==0 && size==std::streamoff(8)+std::streamoff(p.vertices.size())*std::streamoff(sizeof(Vertex)),
                    "invalid mesh size or triangle count: "+file);
            for (const Vertex& v:p.vertices)
                require(std::isfinite(v.p.x) && std::isfinite(v.p.y) && std::isfinite(v.p.z) &&
                        std::isfinite(v.n.x) && std::isfinite(v.n.y) && std::isfinite(v.n.z) &&
                        std::isfinite(v.uv.x) && std::isfinite(v.uv.y) && glm::dot(v.n,v.n)>1e-12f,
                        "invalid vertex in "+file);
            if (nodes[p.node].visual) visualGeometry=true;
            primitives.push_back(std::move(p));
        }
        require(!primitives.empty() && visualGeometry,"no visual primitives");

        ids.clear(); rows=readRows(path(root,"legacy/anchors.tsv"));
        for (const auto& row:rows) {
            require(row.size()==9 && row[0]=="A" && !row[1].empty(),"invalid anchors.tsv row");
            require(ids.insert(std::make_pair(row[1],1)).second,"duplicate anchor "+row[1]);
            Anchor a; a.id=row[1]; a.node=lookup(nodeIds,row[2],"anchor node");
            a.local=vector3(row,3,"anchors.tsv"); a.expected=vector3(row,6,"anchors.tsv");
            require(glm::length(glm::vec3(base[a.node]*glm::vec4(a.local,1))-a.expected)<=1e-4f,
                    "anchor transform mismatch: "+a.id);
            anchors.push_back(a);
        }
        require(anchors.size()>=3,"at least three anchors are required");
        bool noncollinear=false;
        for (size_t i=1;i<anchors.size();++i) for (size_t j=i+1;j<anchors.size();++j)
            if (glm::length(glm::cross(anchors[i].expected-anchors[0].expected,
                                       anchors[j].expected-anchors[0].expected))>1e-5f) noncollinear=true;
        require(noncollinear,"anchors are collinear");

        rows=readRows(path(root,"legacy/controls.tsv"));
        for (const auto& row:rows) {
            if (row[0]=="I") {
                require(row.size()==7 && interaction<0,"invalid or duplicate interaction");
                interaction=lookup(nodeIds,row[1],"interaction node"); axis=vector3(row,2,"controls.tsv");
                require(glm::length(axis)>1e-6f,"zero interaction axis"); axis=glm::normalize(axis);
                minAngle=number(row[5],"controls.tsv"); maxAngle=number(row[6],"controls.tsv");
                require(minAngle<=0 && maxAngle>=0 && minAngle<=maxAngle,"interaction range must contain the default angle 0");
            } else if (row[0]=="H") {
                require(row.size()==4 && probeParent<0,"invalid or duplicate hierarchy probe");
                probeParent=lookup(nodeIds,row[1],"probe parent"); probeDescendant=lookup(nodeIds,row[2],"probe descendant");
                probeAngle=number(row[3],"controls.tsv");
                require(std::abs(probeAngle)>1e-5f && probeParent!=probeDescendant &&
                        descendantOf(probeDescendant,probeParent),"invalid hierarchy probe relationship");
            } else if (row[0]=="V") {
                require(row.size()==8,"invalid view row");
                for (size_t i=2;i<8;++i) number(row[i],"controls.tsv");
            } else require(false,"unknown controls.tsv row");
        }
        if (interaction>=0) {
            bool found=false;
            for (const Primitive& p:primitives) if (nodes[p.node].visual && descendantOf(p.node,interaction)) found=true;
            require(found,"interaction has no visual geometry");
        }
        if (probeParent>=0) {
            bool found=false;
            for (const Primitive& p:primitives) if (descendantOf(p.node,probeDescendant)) found=true;
            require(found,"hierarchy probe descendant has no geometry");
            const glm::mat4 rotation=glm::rotate(glm::mat4(1),glm::radians(probeAngle),glm::vec3(0,1,0));
            const glm::mat4 expected=base[probeParent]*rotation*glm::inverse(base[probeParent])*base[probeDescendant];
            const glm::mat4 actual=worlds(probeParent,glm::vec3(0,1,0),probeAngle)[probeDescendant];
            require(maxDifference(actual,expected)<1e-4f && maxDifference(actual,base[probeDescendant])>1e-5f,
                    "hierarchy motion failed");
        }
    }
public:
    static std::unique_ptr<CampusAsset> loadOptional(const std::string& directory) {
#ifdef _WIN32
        const DWORD attributes=GetFileAttributesA(directory.c_str());
        if (attributes==INVALID_FILE_ATTRIBUTES) {
            const DWORD error=GetLastError();
            if (error==ERROR_FILE_NOT_FOUND || error==ERROR_PATH_NOT_FOUND) return std::unique_ptr<CampusAsset>();
            throw std::runtime_error("Cannot inspect Lide asset directory");
        }
        require((attributes & FILE_ATTRIBUTE_DIRECTORY)!=0,"asset path is not a directory");
#else
        struct stat info;
        if (stat(directory.c_str(),&info)!=0) {
            if (errno==ENOENT || errno==ENOTDIR) return std::unique_ptr<CampusAsset>();
            throw std::runtime_error("Cannot inspect Lide asset directory");
        }
        require(S_ISDIR(info.st_mode),"asset path is not a directory");
#endif
        return std::unique_ptr<CampusAsset>(new CampusAsset(directory));
    }
    ~CampusAsset() { for (const Material& m:materials) if (m.gpuTexture) glDeleteTextures(1,&m.gpuTexture); }
    bool hasInteraction() const { return interaction>=0; }
    float maximumAngle() const { return maxAngle; }
    void validateAngle(float angle) const {
        require(std::isfinite(angle) && angle>=minAngle-1e-5f && angle<=maxAngle+1e-5f,
                "requested angle is outside the interaction range");
    }
    void upload() {
        for (Material& m:materials) {
            m.gpuTexture=m.texture=="-" ? loadTexture("assets/white.ppm") : loadRgbaTexture(path(root,m.texture));
            require(m.gpuTexture!=0 && glGetError()==GL_NO_ERROR,"texture upload failed: "+m.id);
        }
        const auto base=worlds(-1,axis,0);
        std::map<int,std::vector<Vertex> > grouped;
        for (const Primitive& p:primitives) if (nodes[p.node].visual && !dynamic(p))
            appendStaticGeometry(grouped[p.material],p.vertices,base[p.node]);
        for (auto& item:grouped) {
            StaticBatch batch; batch.material=item.first; batch.mesh.reset(new Mesh(item.second));
            require(batch.mesh->vao && batch.mesh->vbo && glGetError()==GL_NO_ERROR,"static batch upload failed");
            staticBatches.push_back(std::move(batch));
        }
        // 交互节点及其后代保留节点局部网格，不能合并入静态缓存。
        for (Primitive& p:primitives) if (nodes[p.node].visual && dynamic(p)) {
            p.mesh.reset(new Mesh(p.vertices));
            require(p.mesh->vao && p.mesh->vbo && glGetError()==GL_NO_ERROR,"mesh upload failed: "+p.id);
        }
    }
    std::string report(float angle) const {
        validateAngle(angle); const auto base=worlds(-1,axis,0),posed=worlds(interaction,axis,angle);
        std::ostringstream out; out << std::setprecision(9) << "{\"status\":\"passed\",\"present\":true,\"placement\":";
        jsonMatrix(out,placement); out << ",\"nodes\":[";
        size_t edges=0; float edgeError=0;
        for (size_t i=0;i<nodes.size();++i) {
            if (i) out << ',';
            const Node& n=nodes[i];
            if (n.parentIndex>=0) { ++edges; edgeError=std::max(edgeError,maxDifference(base[i],base[n.parentIndex]*n.local)); }
            out << "{\"id\":" << json(n.id) << ",\"parent\":" << (n.parentIndex<0 ? "null" : json(n.parent)) << ",\"local_matrix\":";
            jsonMatrix(out,n.local); out << ",\"world_matrix\":"; jsonMatrix(out,base[i]);
            out << ",\"campus_matrix\":"; jsonMatrix(out,placement*posed[i]); out << '}';
        }
        out << "],\"parent_edges_checked\":" << edges << ",\"max_parent_edge_error\":" << edgeError << ",\"anchors\":[";
        for (size_t i=0;i<anchors.size();++i) {
            if (i) out << ',';
            const Anchor& a=anchors[i];
            const glm::vec3 actual=glm::vec3(base[a.node]*glm::vec4(a.local,1));
            out << "{\"id\":" << json(a.id) << ",\"node\":" << json(nodes[a.node].id) << ",\"local\":";
            jsonVector(out,a.local); out << ",\"asset_expected\":"; jsonVector(out,a.expected);
            out << ",\"asset_actual\":"; jsonVector(out,actual); out << ",\"campus_world\":";
            jsonVector(out,glm::vec3(placement*base[a.node]*glm::vec4(a.local,1)));
            out << ",\"error\":" << glm::length(actual-a.expected) << '}';
        }
        out << "],\"interaction\":";
        if (interaction<0) out << "null";
        else { out << "{\"node\":" << json(nodes[interaction].id) << ",\"axis\":"; jsonVector(out,axis);
               out << ",\"range_deg\":[" << minAngle << ',' << maxAngle << "],\"angle_deg\":" << angle << '}'; }
        out << ",\"hierarchy\":";
        if (probeParent<0) out << "null";
        else {
            const auto rotated=worlds(probeParent,glm::vec3(0,1,0),probeAngle);
            out << "{\"parent\":" << json(nodes[probeParent].id) << ",\"descendant\":" << json(nodes[probeDescendant].id)
                << ",\"rotation_deg\":" << probeAngle << ",\"world_before\":";
            jsonMatrix(out,placement*base[probeDescendant]); out << ",\"world_after\":";
            jsonMatrix(out,placement*rotated[probeDescendant]);
            out << ",\"max_abs_delta\":" << maxDifference(base[probeDescendant],rotated[probeDescendant]) << '}';
        }
        size_t staticCount=0,dynamicCount=0; std::vector<bool> batchMaterials(materials.size(),false);
        for (const Primitive& p:primitives) if (nodes[p.node].visual) {
            if (dynamic(p)) ++dynamicCount;
            else { ++staticCount; batchMaterials[p.material]=true; }
        }
        const size_t batchCount=static_cast<size_t>(std::count(batchMaterials.begin(),batchMaterials.end(),true));
        out << ",\"render_cache\":{\"static_primitives\":" << staticCount << ",\"static_batches\":" << batchCount
            << ",\"dynamic_primitives\":" << dynamicCount << ",\"planned_draw_calls_per_pass\":" << batchCount+dynamicCount << "}}\n";
        return out.str();
    }
    void draw(GLuint program,bool depthPass,const glm::mat4& rootMatrix,float angle) const {
        validateAngle(angle); const auto world=worlds(interaction,axis,angle);
        GLint oldActive=0,oldTexture=0,oldVao=0,oldCullMode=0,oldFront=0;
        glGetIntegerv(GL_ACTIVE_TEXTURE,&oldActive); glActiveTexture(GL_TEXTURE0);
        glGetIntegerv(GL_TEXTURE_BINDING_2D,&oldTexture); glGetIntegerv(GL_VERTEX_ARRAY_BINDING,&oldVao);
        glGetIntegerv(GL_CULL_FACE_MODE,&oldCullMode); glGetIntegerv(GL_FRONT_FACE,&oldFront);
        const GLboolean oldCull=glIsEnabled(GL_CULL_FACE);
        const GLint model=glGetUniformLocation(program,"model"),instancing=glGetUniformLocation(program,"useInstancing"),
                    alpha=glGetUniformLocation(program,"alphaCutout"),scale=glGetUniformLocation(program,"textureScale"),
                    surface=glGetUniformLocation(program,"surfaceType"),color=glGetUniformLocation(program,"baseColor");
        GLfloat oldModel[16]={},oldScale[2]={},oldColor[3]={}; GLint oldInstancing=0,oldAlpha=0,oldSurface=0;
        if (model>=0) glGetUniformfv(program,model,oldModel);
        if (instancing>=0) glGetUniformiv(program,instancing,&oldInstancing);
        if (alpha>=0) glGetUniformiv(program,alpha,&oldAlpha);
        if (scale>=0) glGetUniformfv(program,scale,oldScale);
        if (surface>=0) glGetUniformiv(program,surface,&oldSurface);
        if (color>=0) glGetUniformfv(program,color,oldColor);
        glUniform1i(instancing,0); glUniform1i(alpha,0); glUniform2f(scale,1,1); glUniform1i(surface,0);
        // 静态几何只合并为 GPU 缓存；原始节点和网格保留供层级与坐标检查。
        const glm::mat4 staticTransform=rootMatrix*placement;
        glUniformMatrix4fv(model,1,GL_FALSE,glm::value_ptr(staticTransform));
        glFrontFace(glm::determinant(glm::mat3(staticTransform))<0 ? GL_CW : GL_CCW);
        for (const StaticBatch& batch:staticBatches) {
            const Material& material=materials[batch.material];
            if (!depthPass) glUniform3fv(color,1,glm::value_ptr(material.color));
            if (material.doubleSided) glDisable(GL_CULL_FACE); else { glEnable(GL_CULL_FACE); glCullFace(GL_BACK); }
            glBindTexture(GL_TEXTURE_2D,material.gpuTexture); batch.mesh->draw();
        }
        for (const Primitive& primitive:primitives) if (primitive.mesh) {
            const Material& material=materials[primitive.material];
            // 兼容网格仍为节点局部坐标；课程场景根、校园放置、层级矩阵各应用一次。
            const glm::mat4 transform=rootMatrix*placement*world[primitive.node];
            glUniformMatrix4fv(model,1,GL_FALSE,glm::value_ptr(transform));
            if (!depthPass) glUniform3fv(color,1,glm::value_ptr(material.color));
            glFrontFace(glm::determinant(glm::mat3(transform))<0 ? GL_CW : GL_CCW);
            if (material.doubleSided) glDisable(GL_CULL_FACE); else { glEnable(GL_CULL_FACE); glCullFace(GL_BACK); }
            glBindTexture(GL_TEXTURE_2D,material.gpuTexture); primitive.mesh->draw();
        }
        if (model>=0) glUniformMatrix4fv(model,1,GL_FALSE,oldModel);
        if (instancing>=0) glUniform1i(instancing,oldInstancing);
        if (alpha>=0) glUniform1i(alpha,oldAlpha);
        if (scale>=0) glUniform2fv(scale,1,oldScale);
        if (surface>=0) glUniform1i(surface,oldSurface);
        if (color>=0) glUniform3fv(color,1,oldColor);
        glBindVertexArray(static_cast<GLuint>(oldVao)); glBindTexture(GL_TEXTURE_2D,static_cast<GLuint>(oldTexture));
        glActiveTexture(static_cast<GLenum>(oldActive)); glCullFace(static_cast<GLenum>(oldCullMode));
        glFrontFace(static_cast<GLenum>(oldFront)); if (oldCull) glEnable(GL_CULL_FACE); else glDisable(GL_CULL_FACE);
    }
};

#endif
