#include <bits/stdc++.h>
using namespace std;
typedef long long LL;
typedef std::pair<int, int> PII;
typedef std::pair<LL, LL> PLL;
const int N = 100010, M = 200010;

int t, n, m, res, cnt;
int p[N];
unordered_map<string, int> mp;
string v[N];
vector<tuple<int, int, int>> paths;


struct Edges {
    int u, v, w;
    bool operator< (const Edges& e) {
        return w < e.w;
    };
} edges[M];

int find(int x) {
    if (x != p[x]) {
        p[x] = find(p[x]);
    }
    return p[x];
}

void kruskal() {
    int res = 0, cnt = 0;
    
    for (int i = 0; i < m; i++) {
        int a = edges[i].u, b = edges[i].v, c = edges[i].w;
        if (find(a) != find(b)) {
            p[find(a)] = find(b);
            res += c;
            cnt++;
            paths.push_back({a, b, c});
        }
    }


    if (cnt != n - 1) {
        std::cout << "-1";
    } else {
        std::cout << res << "\n";
        for (auto &[a, b, c] : paths) {
            if (a > b) {
                swap(a, b);
            }
            std::cout << v[a] << " " << v[b] << " " << c << "\n";
        }
    }
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::cin >> n;
    for (int i = 1; i <= n; i++) {
        std::cin >> v[i];
        p[i] = i;
        mp[v[i]] = i;
    }

    std::cin >> m;
    for (int i = 0; i < m; i++) {
        string a, b;
        int u, v, w;
        std::cin >> a >> b >> w;
        u = mp[a], v = mp[b];
        edges[i] = {u, v, w};
    }

    sort(edges, edges + m);
    kruskal();
    return 0;
}
