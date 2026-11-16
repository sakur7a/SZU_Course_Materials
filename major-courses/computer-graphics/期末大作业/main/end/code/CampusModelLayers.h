#ifndef SZU_CAMPUS_MODEL_LAYERS_H
#define SZU_CAMPUS_MODEL_LAYERS_H

// 各地点共享坐标、网格格式及材质索引，每栋的每一层仍可单独替换。
// 在 Mesh、readMesh 与 loadTexture 定义后包含，不创建全局 OpenGL 对象。
struct ModelLayerSpec {
    std::string id,material,mesh,texture;
    glm::vec3 color;
};

inline std::vector<ModelLayerSpec> readModelLayerSpecs(const std::string& index) {
    std::ifstream file(index);
    if (!file) throw std::runtime_error("Cannot read campus model index: "+index);
    std::vector<ModelLayerSpec> result;std::string line;
    while (std::getline(file,line)) {
        std::istringstream row(line);std::string kind;
        if (!(row>>kind) || kind!="L") continue;
        ModelLayerSpec layer;
        if (!(row>>layer.id>>layer.material>>layer.mesh>>layer.texture
                 >>layer.color.r>>layer.color.g>>layer.color.b))
            throw std::runtime_error("Invalid campus model layer: "+index+": "+line);
        result.push_back(layer);
    }
    if (result.empty()) throw std::runtime_error("Empty campus model index: "+index);
    return result;
}

class CampusModelLayers {
    struct Layer {
        std::unique_ptr<Mesh> mesh;
        GLuint texture;
        glm::vec3 color;
        int cutout,surface;
        Layer(const ModelLayerSpec& spec,GLuint tex)
            : mesh(new Mesh(readMesh("assets/"+spec.mesh))),texture(tex),color(spec.color),
              cutout(spec.texture.size()>5 && spec.texture.substr(spec.texture.size()-5)==".rgba" ? 1 : 0),
              surface(spec.material=="water" ? 1 : spec.material=="foliage" ? 2 : 0) {}
    };
    std::vector<Layer> layers;
    std::map<std::string,GLuint> textures;
public:
    explicit CampusModelLayers(const std::string& index,const std::string& excludedId="") {
        const std::vector<ModelLayerSpec> specs=readModelLayerSpecs(index);
        layers.reserve(specs.size());
        try {
            for (const ModelLayerSpec& spec:specs) {
                // 新门模型成功载入后只替换对应地点；门区视角和其他地点不变。
                if (!excludedId.empty() && spec.id==excludedId) continue;
                std::string filename=spec.texture;
                if (filename=="-") {
                    if (spec.material=="sign")
                        throw std::runtime_error("Campus sign texture is missing: "+spec.id);
                    if (spec.material=="stone" || spec.material=="roof") filename="concrete.ppm";
                    else if (spec.material=="glass") filename="glass.ppm";
                    else filename="white.ppm";
                }
                auto found=textures.find(filename);
                if (found==textures.end()) {
                    const bool rgba=filename.size()>5 && filename.substr(filename.size()-5)==".rgba";
                    const GLuint texture=rgba ? loadRgbaTexture("assets/"+filename) : loadTexture("assets/"+filename);
                    found=textures.insert(std::make_pair(filename,texture)).first;
                }
                layers.emplace_back(spec,found->second);
            }
        } catch (...) {
            for (const auto& item:textures) glDeleteTextures(1,&item.second);
            throw;
        }
    }
    ~CampusModelLayers() {
        for (const auto& item:textures) glDeleteTextures(1,&item.second);
    }
    CampusModelLayers(const CampusModelLayers&)=delete;
    CampusModelLayers& operator=(const CampusModelLayers&)=delete;

    void draw(GLuint program,bool depthPass,const glm::mat4& root) const {
        glUniform1i(glGetUniformLocation(program,"useInstancing"),0);
        glUniformMatrix4fv(glGetUniformLocation(program,"model"),1,GL_FALSE,glm::value_ptr(root));
        glUniform1i(glGetUniformLocation(program,"alphaCutout"),0);
        glUniform2f(glGetUniformLocation(program,"textureScale"),1,1);
        glUniform1i(glGetUniformLocation(program,"surfaceType"),0);
        const GLint colorLocation=depthPass ? -1 : glGetUniformLocation(program,"baseColor");
        glActiveTexture(GL_TEXTURE0);
        for (const Layer& layer:layers) {
            glUniform1i(glGetUniformLocation(program,"alphaCutout"),layer.cutout);
            glUniform1i(glGetUniformLocation(program,"surfaceType"),layer.surface);
            if (!depthPass) glUniform3fv(colorLocation,1,glm::value_ptr(layer.color));
            glBindTexture(GL_TEXTURE_2D,layer.texture);layer.mesh->draw();
        }
    }
};

#endif
