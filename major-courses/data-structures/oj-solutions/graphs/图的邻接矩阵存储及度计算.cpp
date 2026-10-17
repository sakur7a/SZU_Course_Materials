#include <bits/stdc++.h>
using namespace std;
const int N = 1010;

unordered_map<string, int> toIdx;
int t, n, m;
string v[N];
int g[N][N];

void solve() {
    memset(g, 0, sizeof(g));
    char op;
    std::cin >> op >> n;
    for (int i = 0; i < n; i++) {
        std::cin >> v[i];
        toIdx[v[i]] = i;
    }

    std::cin >> m;
    while (m--) {
        string u, v;
        std::cin >> u >> v;
        int a = toIdx[u], b = toIdx[v];
        g[a][b] = 1;
        if (op == 'U') {
            g[b][a] = 1;
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j) {
                std::cout << " ";
            }
            std::cout << g[i][j];
        }
        std::cout << "\n";
    }

    vector<string> isolated;
    if (op == 'U') {
        for (int i = 0; i < n; i++) {
            int degree = 0;
            for (int j = 0; j < n; j++) {
                degree += g[i][j];
            }
            if (degree) {
                std::cout << v[i] << ": " << degree << "\n";
            } else {
                isolated.push_back(v[i]);
            }
        }
    } else {
        for (int i = 0; i < n; i++) {
            int outDegree = 0, inDegree = 0;
            for (int j = 0; j < n; j++) {
                outDegree += g[i][j];
                inDegree += g[j][i];
            }

            int tot = outDegree + inDegree;
            if (tot) {
                std::cout << v[i] << ": " << outDegree << " " << inDegree << " " << tot << "\n";
            } else {
                isolated.push_back(v[i]);
            }
        }
    }

    for (auto s : isolated) {
        std::cout << s << "\n";
    }
}

int main() {
    std::cin >> t;
    while (t--) {
        solve();
    }
    return 0;
}