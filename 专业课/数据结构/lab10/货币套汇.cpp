#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>

using namespace std;

void solve() {
    int n, m;
    // 读取货币数量和兑换率数量
    if (!(cin >> n >> m)) return;

    // 使用 map 将货币名称映射为整数索引 0 ~ n-1
    map<string, int> nameToId;
    for (int i = 0; i < n; ++i) {
        string name;
        cin >> name;
        nameToId[name] = i;
    }

    // 初始化邻接矩阵
    // dist[i][j] 表示从货币 i 到货币 j 的最大汇率
    vector<vector<double>> dist(n, vector<double>(n, 0.0));

    // 自身到自身的汇率初始化为 1.0
    for (int i = 0; i < n; ++i) {
        dist[i][i] = 1.0;
    }

    // 读取 m 条兑换规则
    for (int i = 0; i < m; ++i) {
        string s1, s2;
        double r;
        cin >> s1 >> r >> s2;
        int u = nameToId[s1];
        int v = nameToId[s2];
        // 这是一个有向图，更新 u -> v 的汇率
        // 如果存在多条重边，保留汇率最高的那条
        if (r > dist[u][v]) {
            dist[u][v] = r;
        }
    }

    // Floyd-Warshall 算法
    // k 为中间节点
    for (int k = 0; k < n; ++k) {
        // i 为起点
        for (int i = 0; i < n; ++i) {
            // j 为终点
            for (int j = 0; j < n; ++j) {
                // 如果经过 k 的路径汇率更高，则更新
                // 注意这里是乘法，因为是汇率兑换
                if (dist[i][k] * dist[k][j] > dist[i][j]) {
                    dist[i][j] = dist[i][k] * dist[k][j];
                }
            }
        }
    }

    // 检查是否存在对角线元素大于 1.0 的情况
    // 即是否存在从某货币出发回到该货币，价值增加
    bool arbitrage = false;
    for (int i = 0; i < n; ++i) {
        if (dist[i][i] > 1.0) {
            arbitrage = true;
            break;
        }
    }

    if (arbitrage) {
        cout << "YES" << endl;
    } else {
        cout << "NO" << endl;
    }
}

int main() {
    // 优化 IO 效率
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int t;
    // 读取测试组数
    if (cin >> t) {
        while (t--) {
            solve();
        }
    }
    return 0;
