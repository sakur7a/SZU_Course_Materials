#include <bits/stdc++.h>
using namespace std;
typedef long long LL;
typedef std::pair<int, int> PII;
typedef std::pair<LL, LL> PLL;
const int N = 1010;

int t, n, m;

void dfs(int idx, vector<bool>& visited, const vector<vector<int>>& adj) {
    visited[idx] = true;
    int n = adj.size();
    for (int i = 1; i <= n; i++) {
        if (adj[idx][i] == 1 && !visited[i]) {
            dfs(i, visited, adj);
        }
    }
}


int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::cin >> t;
    for (int w = 0; w < t; w++) {
        if (w > 0) {
            std::cout << "\n";
        }
        unordered_map<string, int> mp;
        std::cin >> n;
        vector<vector<int>> g(n + 1, vector<int>(n + 1, 0));
        vector<string> v(n + 1);

        for (int i = 1; i <= n; i++) {
            std::cin >> v[i];
            mp[v[i]] = i; 
        }

        std::cin >> m;
        while (m--) {
            string a, b;
            std::cin >> a >> b;
            g[mp[a]][mp[b]] = 1, g[mp[b]][mp[a]] = 1;
        }

        for (int i = 1; i <= n; i++) {
            if (i > 1) {
                std::cout << " ";
            }
            std::cout << v[i];
        }
        std::cout << "\n";

        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                if (j > 1) {
                    std::cout << " ";
                }
                std::cout << g[i][j];
            }
            std::cout << "\n";
        }

        int count = 0;
        vector<bool> visited(n, false); 
        for (int i = 1; i <= n; i++) {
            if (!visited[i]) { // 找到未访问的顶点，开始新的连通分量
                count++;
                dfs(i, visited, g); 
            }
        }
        cout << count << "\n";
    }
  
    return 0;
}
