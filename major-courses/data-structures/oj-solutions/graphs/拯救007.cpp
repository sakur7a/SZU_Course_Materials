#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

// 定义鳄鱼坐标结构体
struct Point {
    int x, y;
};

int N;
double D;
vector<Point> crocs;
vector<bool> visited; // 记录鳄鱼是否已被访问

// 判断能否从鳄鱼 u 跳到 鳄鱼 v
// 两点间距离公式：sqrt((x1-x2)^2 + (y1-y2)^2)
// 为了避免开方带来的精度损失，比较平方值：dist^2 <= D^2
bool canJump(int u, int v) {
    double dist2 = pow(crocs[u].x - crocs[v].x, 2) + pow(crocs[u].y - crocs[v].y, 2);
    return dist2 <= D * D;
}

// 判断能否从池心岛跳到鳄鱼 u
// 岛半径 7.5，起跳点在岛边缘，所以能跳到的最大半径为 D + 7.5
bool firstJump(int u) {
    double dist2 = pow(crocs[u].x, 2) + pow(crocs[u].y, 2);
    double maxDist = D + 7.5;
    return dist2 <= maxDist * maxDist;
}

// 判断能否从鳄鱼 u 跳上岸
// 池子范围是 -50 到 50
// 只要鳄鱼到上下左右任一边界的距离 <= D 即可
// 即：abs(x) + D >= 50 或 abs(y) + D >= 50
bool reachBank(int u) {
    return (abs(crocs[u].x) + D >= 50) || (abs(crocs[u].y) + D >= 50);
}

// 深度优先搜索
bool dfs(int u) {
    visited[u] = true;
    
    // 如果当前鳄鱼能直接跳上岸，返回成功
    if (reachBank(u)) return true;
    
    // 否则尝试跳到其他未访问的鳄鱼
    for (int v = 0; v < N; ++v) {
        if (!visited[v] && canJump(u, v)) {
            // 如果从 v 能逃脱，则说明从 u 也能逃脱
            if (dfs(v)) return true;
        }
    }
    
    return false;
}

int main() {
    // 优化IO效率
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    if (cin >> N >> D) {
        // 特殊情况：如果在岛上直接能跳上岸
        // 岛边缘距离岸边最近距离为 50 - 7.5 = 42.5
        if (D >= 42.5) {
            cout << "Yes" << endl;
            return 0;
        }

        crocs.resize(N);
        visited.assign(N, false);
        
        for (int i = 0; i < N; ++i) {
            cin >> crocs[i].x >> crocs[i].y;
        }

        bool possible = false;
        
        // 遍历每条鳄鱼，如果它能从岛上跳到，就以它为起点开始DFS
        for (int i = 0; i < N; ++i) {
            if (!visited[i] && firstJump(i)) {
                if (dfs(i)) {
                    possible = true;
                    break; // 只要找到一条路即可退出
                }
            }
        }

        if (possible) cout << "Yes" << endl;
        else cout << "No" << endl;
    }
    return 0;
}