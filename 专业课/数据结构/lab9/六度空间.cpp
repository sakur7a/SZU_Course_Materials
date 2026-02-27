#include <bits/stdc++.h>
using namespace std;
typedef long long LL;
typedef std::pair<int, int> PII;
typedef std::pair<LL, LL> PLL;
const int N = 1010, M = 33 * N;

int t, n, m;
vector<int> g[N];
bool st[N];


void bfs(int u) {
    queue<PII> q;
    memset(st, 0, sizeof(st));
    int cnt = 1;
    st[u] = true;
    q.push({u, 0});

    while (q.size()) {
        auto &[o, d] = q.front();
        q.pop();

        if (d == 6) {
            continue;
        }

        for (auto &v : g[o]) {
            if (!st[v]) {
                st[v] = true;
                cnt++;
                q.push({v, d + 1});
            }
        }
    }
    double percentage = (double)cnt / n * 100.0;
    cout << u << ": " << fixed << setprecision(2) << percentage << "%" << endl;
}


int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::cin >> n >> m;

    for (int i = 0; i < m; i++) {
        int a, b;
        std::cin >> a >> b;
        g[a].push_back(b), g[b].push_back(a);
    }

    for (int i = 1; i <= n; i++) {
        bfs(i);
    }
    return 0;
}
