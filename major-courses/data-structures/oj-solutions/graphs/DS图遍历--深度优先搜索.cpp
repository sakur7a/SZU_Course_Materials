#include <bits/stdc++.h>
using namespace std;
typedef long long LL;
typedef std::pair<int, int> PII;
typedef std::pair<LL, LL> PLL;
const int N = 1010;

int t, n, m;
int g[N][N];
bool st[N];

void dfs(int u) {
    st[u] = true;
    cout << u << " ";       

    for (int v = 0; v < n; v++) {
        // 若v与u相连，且v未被访问，则递归访问v
        if (g[u][v] && !st[v]) {
            dfs(v);
        }
    }
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::cin >> t;
    while (t--) {
        memset(st, false, sizeof(st));
        std::cin >> n;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                std::cin >> g[i][j];
            }
        }
        // 注意分离点
        for (int i = 0; i < n; i++) {
            if (!st[i]) {
                dfs(i);
            }
        }
        std::cout << "\n";
    }
    return 0;
}
