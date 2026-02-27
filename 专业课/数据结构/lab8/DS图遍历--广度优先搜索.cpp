#include <bits/stdc++.h>
using namespace std;
typedef long long LL;
typedef std::pair<int, int> PII;
typedef std::pair<LL, LL> PLL;
const int N = 1010;

int t, n, m;
int g[N][N];
bool st[N];

void bfs(int u) {
    st[u] = true;
    queue<int> q;
    q.push(u);
    std::cout << u << " ";
    while (q.size()) {
        int t = q.front();
        q.pop();

        for (int v = 0; v < n; v++) {
            if (g[t][v] && !st[v]) {
                st[v] = true;
                std::cout << v << " ";
                q.push(v);
            }
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
        for (int i = 0; i < n; i++) {
            if (!st[i]) {
                bfs(i);
            }
        }
        std::cout << "\n";
    }
  
    return 0;
}
