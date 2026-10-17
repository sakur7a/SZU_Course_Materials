#include <iostream>
#include <vector>
#include <string>
#include <queue>
#include <algorithm>

using namespace std;

// 顶点类
class Vertex {
public:
    int indexNo;
    bool hasEnterQueue;
    int early;
    int later;

    Vertex(int indexNo) {
        this->indexNo = indexNo;
        this->hasEnterQueue = false;
        early = -1;
        // 初始化为一个较大的数，作为无穷大
        later = 0x7FFFFFFF; 
    }
    void updateEarly(int parentEarly, int edgeValue) {
        int newEarly = parentEarly + edgeValue;
        if (newEarly > this->early)
            this->early = newEarly;
    }
    void updateLater(int childLater, int edgeValue) {
        int newLater = childLater - edgeValue;
        if (newLater < this->later)
            this->later = newLater;
    }
};

// 图类
class Graph {
public:
    vector<Vertex> vertexes;
    vector<vector<int> > adjMat;
    int n;
public:
    void readVertexes() {
        // 读取顶点数
        if (!(cin >> n)) return;
        
        // 初始化顶点数组
        for(int i=0; i<n; ++i) {
            Vertex v(i);
            this->vertexes.push_back(v);
        }
        
        // 初始化邻接矩阵，全部置0
        // resize 调整大小，vector<int>(n, 0) 创建一行n个0
        adjMat.resize(n, vector<int>(n, 0));
    }

    void readAdjMatrix() {
        int edges;
        cin >> edges;
        int s, t, w;  // s源顶点编号，t目的顶点编号，w边长
        for(int i=0; i<edges; ++i) {
            cin >> s >> t >> w;
            adjMat[s][t] = w;
        }
    }

    // 更新最早开始时间（前向）
    void updateEarly(int parentNo, queue<int>& earlyQue) {
        int parentEarly = vertexes[parentNo].early;

        for(int j=0; j<n; ++j) {
            int edgeValue = adjMat[parentNo][j];
            if (edgeValue == 0) continue;  // 无边连接

            Vertex& child = vertexes[j];
            
            // 记录更新前的值，用于判断是否发生变化
            int oldEarly = child.early;
            
            child.updateEarly(parentEarly, edgeValue); // 尝试更新

            // 如果值变大（被更新了），则需要将子节点加入队列进行后续传播
            // 这里使用类似SPFA的逻辑，即使节点之前进过队列，如果值更新了也需要重新处理
            if(child.early > oldEarly) {
                if(!child.hasEnterQueue) {
                    child.hasEnterQueue = true;
                    earlyQue.push(j);
                }
            } else if (oldEarly == -1 && !child.hasEnterQueue) {
                // 处理第一次被访问的情况（即使值没变大，比如从-1变到初始值）
                child.hasEnterQueue = true;
                earlyQue.push(j);
            }
        }
    }

    // 更新最迟开始时间（后向）
    void updateLater(int childNo, queue<int>& laterQue) {
        int childLater = vertexes[childNo].later;
        
        // 遍历所有可能的父节点
        for(int i=0; i<n; ++i) {
            // 检查是否存在边 i -> childNo
            int edgeValue = adjMat[i][childNo];
            if (edgeValue == 0) continue;

            Vertex& parent = vertexes[i];
            int oldLater = parent.later;
            
            parent.updateLater(childLater, edgeValue);

            // 如果最迟时间变小了（被更新了），加入队列
            if (parent.later < oldLater) {
                if (!parent.hasEnterQueue) {
                    parent.hasEnterQueue = true;
                    laterQue.push(i);
                }
            }
        }
    }

    int getRoot() {
        // 获取入度为0的顶点 (仅用于参考，实际findEarly逻辑已增强)
        for(int j=0; j<n; ++j) {
            int i=0;
            for(; i<n && adjMat[i][j] == 0; ++i);
            if (i>=n) return j; 
        }
        return -1;
    }
    
    int getLeaf() {
        // 获取出度为0的顶点 (任意一个)
        for(int i=0; i<n; ++i) {
            bool isLeaf = true;
            for(int j=0; j<n; ++j) {
                if (adjMat[i][j] != 0) {
                    isLeaf = false;
                    break;
                }
            }
            if(isLeaf) return i;
        }
        return -1;
    }

    void printEarlyLater(bool isEarly) {
        for(int i=0; i<n; ++i) {
            Vertex& v = vertexes[i];
            if (isEarly)
                cout << v.early << " ";
            else {
                cout << v.later << " ";
            }
        }
        cout << endl;
    }

    void findEarly() {
        // 初始化所有入度为0的节点（Roots）的early为0，并加入队列
        queue<int> que;
        
        for(int i=0; i<n; ++i) {
            bool isRoot = true;
            for(int k=0; k<n; ++k) {
                if(adjMat[k][i] != 0) {
                    isRoot = false;
                    break;
                }
            }
            if(isRoot) {
                vertexes[i].early = 0;
                vertexes[i].hasEnterQueue = true;
                que.push(i);
            }
        }

        while(!que.empty()) {
            int p = que.front();
            que.pop();
            // 关键：出队后重置标志，允许节点在后续路径优化时再次入队（SPFA逻辑）
            vertexes[p].hasEnterQueue = false;

            updateEarly(p, que);
        }

        printEarlyLater(true);
    }
    
    void clearEnterQueue() {
        for(int i=0; i<n; ++i) {
            vertexes[i].hasEnterQueue = false;
        }
    }
    
    void findLater() {
        clearEnterQueue();
        
        // 1. 找到整个工程的结束时间（即所有节点中最大的early时间）
        int maxEarly = 0;
        for(int i=0; i<n; ++i) {
            if(vertexes[i].early > maxEarly) maxEarly = vertexes[i].early;
        }
        
        // 2. 将所有汇点（出度为0的节点）的later初始化为结束时间，并加入队列
        queue<int> que;
        for(int i=0; i<n; ++i) {
            bool isLeaf = true;
            for(int j=0; j<n; ++j) {
                if(adjMat[i][j] != 0) {
                    isLeaf = false;
                    break;
                }
            }
            if(isLeaf) {
                vertexes[i].later = maxEarly;
                vertexes[i].hasEnterQueue = true;
                que.push(i);
            } else {
                vertexes[i].later = 0x7FFFFFFF; // 初始化为无穷大
            }
        }
        
        // 3. 反向传播计算最迟开始时间
        while(!que.empty()) {
            int p = que.front();
            que.pop();
            vertexes[p].hasEnterQueue = false; // 允许重新入队

            updateLater(p, que);
        }
        
        printEarlyLater(false);
    }

    void main() {
        readVertexes();
        readAdjMatrix();
        findEarly();
        findLater();
    }
};

int main() {
    // 只有一个测试用例，直接运行
    Graph g;
    g.main();
    return 0;
}