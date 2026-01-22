#include <bits/stdc++.h>
using namespace std;
typedef long long LL;
typedef std::pair<int, int> PII;
typedef std::pair<LL, LL> PLL;
const int N = 1010, INF = 0x3f3f3f3f;

struct Edge {
    int e, l, p;
};

int t, n, m, s, d;
int dist[N], cost[N];
bool st[N];
vector<vector<Edge>> adj(N);

void dijkstra() {
    dist[s] = 0, cost[s] = 0;
    for (int i = 0; i < n; i++) {
        int t = -1;
        for (int j = 0; j < n; j++) {
            if (!st[j] && (t == -1 || dist[t] > dist[j])) {
                t = j;
            }
        }
        if (t == -1) {
            break;
        }
        st[t] = true;

        for (auto& edge : adj[t]) {
            int v = edge.e;
            if (!st[v]) {
                int nd = dist[t] + edge.l;
                int np = cost[t] + edge.p;
                if (nd < dist[v]) {
                    dist[v] = nd;
                    cost[v] = np;
                } else if (nd == dist[v]) {
                    cost[v] = min(cost[v], np);
                }
            }
        }
    }

    cout << dist[d] << " " << cost[d] << "\n";

}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    memset(dist, 0x3f, sizeof(dist));
    memset(cost, 0x3f, sizeof(cost));

    std::cin >> n >> m >> s >> d;

    for (int i = 0; i < m; i++) {
        int a, b, c, p;
        std::cin >> a >> b >> c >> p;
        adj[a].push_back({b, c, p});
        adj[b].push_back({a, c, p});
    }
    dijkstra();
    return 0;
}
