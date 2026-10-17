#include <bits/stdc++.h>
using namespace std;
typedef long long LL;
typedef std::pair<int, int> PII;
typedef std::pair<LL, LL> PLL;
const int N = 100010;


int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int N, M;
    cin >> N >> M;
    vector<vector<int>> adj(N + 1);  

    for (int i = 0; i < M; ++i) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    int K;
    cin >> K;
    while (K--) {
        int target;
        cin >> target;
        vector<int> dist(N + 1, -1);  
        queue<int> q;

        // BFS
        dist[target] = 0;
        q.push(target);

        while (!q.empty()) {
            int u = q.front();
            q.pop();
            for (int neighbor : adj[u]) {
                if (dist[neighbor] == -1) { 
                    dist[neighbor] = dist[u] + 1;
                    q.push(neighbor);
                }
            }
        }

        bool all_reachable = true;
        int sum_d = 0;
        for (int j = 1; j <= N; ++j) {
            if (j == target) continue;  
            if (dist[j] == -1) {        
                all_reachable = false;
                break;
            }
            sum_d += dist[j];
        }

        double cc = 0.0;
        if (all_reachable) {
            cc = (N - 1.0) / sum_d;  
        }

        printf("Cc(%d)=%.2f\n", target, cc);
    }

  
    return 0;
}
