#ifndef SZU_CAMPUS_GATES_H
#define SZU_CAMPUS_GATES_H

// 门区索引维护逐门视角；网格与材质由各地点共用的分层加载器处理。
struct GateView {
    std::string id,name;
    glm::vec3 eye,target;
    glm::vec2 center;
};

inline std::vector<GateView> readGateViews(const std::string& index="assets/campus-gates.tsv") {
    std::ifstream file(index);
    if (!file) throw std::runtime_error("Cannot read campus entrance view index: "+index);
    std::vector<GateView> result;std::string line;
    while (std::getline(file,line)) {
        std::istringstream row(line);std::string kind;
        if (!(row>>kind) || kind!="G") continue;
        GateView gate;
        if (!(row>>gate.id>>gate.name>>gate.eye.x>>gate.eye.y>>gate.eye.z
                 >>gate.target.x>>gate.target.y>>gate.target.z>>gate.center.x>>gate.center.y))
            throw std::runtime_error("Invalid campus gate view: "+line);
        result.push_back(gate);
    }
    return result;
}

#endif
