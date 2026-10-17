#include <bits/stdc++.h>
using namespace std;
typedef long long LL;
typedef std::pair<int, int> PII;
typedef std::pair<LL, LL> PLL;
const int N = 1010;

int t, n, m;
int dir[4][2] =  {{-1, 0}, {1, 0}, {0, -1}, {0, 1}}; 
int g[N][N];
bool st[N][N];

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::cin >> t;
    while (t--) {
        queue<PII> q;
        memset(st, false, sizeof(st));

        std::cin >> n >> m;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                std::cin >> g[i][j];
            }
        }

        // 标记外部 0
        for (int j = 0; j < m; j++) {
            if (g[0][j] == 0 && !st[0][j]) {
                st[0][j] = true;
                q.push({0, j});
            }
            if (g[n - 1][j] == 0 && !st[n - 1][j]) {
                st[n - 1][j] = true;
                q.push({n - 1, j});
            }
        }

        for (int i = 1; i < n - 1; ++i) {
            if (g[i][0] == 0 && !st[i][0]) {
                st[i][0] = true;
                q.push({i, 0});
            }

            if (g[i][m - 1] == 0 && !st[i][m - 1]) {
                st[i][m - 1] = true;
                q.push({i, m - 1});
            }
        }

        while (q.size()) {
            auto [i, j] = q.front();
            q.pop();

            for (auto &d : dir) {
                int ni = i + d[0], nj = j + d[1];
                 if (ni >= 0 && ni < n && nj >= 0 && nj < m && g[ni][nj] == 0 && !st[ni][nj]) {
                    st[ni][nj] = true;
                    q.push({ni, nj});
                }
            }
        }

        int ans = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (g[i][j] == 0 && !st[i][j]) {
                    ans++;
                }
            }
        }

        std::cout << ans << "\n";
    }
  
    return 0;
}
