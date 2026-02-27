#include <bits/stdc++.h>
using namespace std;
typedef long long LL;
typedef std::pair<int, int> PII;
typedef std::pair<LL, LL> PLL;
const int N = 1010, INF = 0x3f3f3f3f;

int t, n, m;
unordered_map<string, int> mp;
string v[N];
string start;
int g[N][N];
int dist[N], parents[N];
bool st[N];
vector<tuple<int, int, int>> paths;

void prim() { // 题目保证有解
    int res = 0;
    for (int i = 0; i < n; i++) {
        int t = -1;
        for (int j = 0; j < n; j++) {
            if (i == 0) {
                t = mp[start];
                break;
            }
            if (!st[j] && (t == -1 || dist[t] > dist[j])) {
                t = j;
            }
        }
       
        if (i) {
            paths.push_back({parents[t], t, dist[t]});
            res += dist[t];
        }
        st[t] = true;

        for (int j = 0; j < n; j++) {
            if (!st[j] && dist[j] > g[j][t]) {
                dist[j] = g[j][t];
                parents[j] = t;
            }
        }
    }
    std::cout << res <<"\n";
    for (auto &[a, b, c] : paths) {
        std::cout << v[a] << " " << v[b] << " " << c << "\n";
    }
}   

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    memset(g, 0x3f, sizeof(g));
    memset(dist, 0x3f, sizeof(dist));

    std::cin >> n;
    for (int i = 0; i < n; i++) {
        std::cin >> v[i];
        mp[v[i]] = i;
    }

    std::cin >> m;
    while (m--) {
        string a, b;
        int u, v, w;
        std::cin >> a >> b >> w;
        u = mp[a], v = mp[b];
        g[u][v] = g[v][u] = min(w, g[u][v]);
    }
    

    std::cin >> start;
    prim();
    return 0;
}
