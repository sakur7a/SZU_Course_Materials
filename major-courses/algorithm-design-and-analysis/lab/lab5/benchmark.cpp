#include <bits/stdc++.h>
using namespace std;

// =====================================================================
// 并查集算法
// =====================================================================
int run_unionfind(int n, int m, const vector<int>& eu, const vector<int>& ev,
                  const vector<vector<pair<int,int>>>& adj_full, const vector<bool>& skip) {
    vector<bool> is_bridge(m, true);
    for (int i = 0; i < m; i++) if (skip[i]) is_bridge[i] = false;

    vector<int> parent(n, -1), parent_edge(n, -1), depth(n, -1);
    vector<bool> visited(n, false);
    vector<int> bu, bv, be;

    vector<int> su, sd, sp, spe;
    for (int start = 0; start < n; start++) {
        if (visited[start]) continue;
        su.push_back(start); sd.push_back(0); sp.push_back(-1); spe.push_back(-1);
        while (!su.empty()) {
            int u = su.back(); su.pop_back();
            int d = sd.back(); sd.pop_back();
            int p = sp.back(); sp.pop_back();
            int pe = spe.back(); spe.pop_back();
            if (visited[u]) continue;
            visited[u] = true;
            depth[u] = d; parent[u] = p; parent_edge[u] = pe;
            for (auto [v, ei] : adj_full[u]) {
                if (!visited[v]) {
                    su.push_back(v); sd.push_back(d+1); sp.push_back(u); spe.push_back(ei);
                } else if (v != p && depth[v] < d) {
                    bu.push_back(u); bv.push_back(v); be.push_back(ei);
                }
            }
        }
    }

    for (int i = 0; i < (int)be.size(); i++) is_bridge[be[i]] = false;

    vector<int> ufp(n), ufr(n, 0);
    for (int i = 0; i < n; i++) ufp[i] = i;
    auto find = [&](int x) -> int {
        int r = x;
        while (ufp[r] != r) r = ufp[r];
        while (ufp[x] != x) { int nx = ufp[x]; ufp[x] = r; x = nx; }
        return r;
    };

    for (int i = 0; i < (int)bu.size(); i++) {
        int x = bu[i], y = bv[i];
        while (true) {
            int fx = find(x), fy = find(y);
            if (fx == fy) break;
            if (depth[fx] > depth[fy]) {
                int eidx = parent_edge[fx];
                if (eidx != -1) is_bridge[eidx] = false;
                ufp[fx] = find(parent[fx]);
            } else if (depth[fy] > depth[fx]) {
                int eidx = parent_edge[fy];
                if (eidx != -1) is_bridge[eidx] = false;
                ufp[fy] = find(parent[fy]);
            } else {
                int eidx = parent_edge[fx];
                if (eidx != -1) is_bridge[eidx] = false;
                ufp[fx] = find(parent[fx]);
            }
        }
    }

    int count = 0;
    for (int i = 0; i < m; i++) if (is_bridge[i]) count++;
    return count;
}

// =====================================================================
// Tarjan算法（迭代版）
// =====================================================================
int run_tarjan(int n, int m, const vector<vector<pair<int,int>>>& adj_full, const vector<bool>& skip) {
    vector<int> disc(n, -1), low(n, -1);
    vector<bool> is_bridge(m, false);
    int timer = 0;

    struct Frame { int u; int pe; size_t ci; bool entered; };

    for (int start = 0; start < n; start++) {
        if (disc[start] != -1) continue;
        vector<Frame> stk;
        stk.push_back({start, -1, 0, false});
        while (!stk.empty()) {
            auto& f = stk.back();
            if (!f.entered) {
                disc[f.u] = low[f.u] = timer++;
                f.entered = true;
                f.ci = 0;
            }
            bool pushed = false;
            while (f.ci < adj_full[f.u].size()) {
                auto [v, eidx] = adj_full[f.u][f.ci];
                f.ci++;
                if (eidx == f.pe) continue;
                if (disc[v] == -1) {
                    stk.push_back({v, eidx, 0, false});
                    pushed = true;
                    break;
                } else {
                    low[f.u] = min(low[f.u], disc[v]);
                }
            }
            if (pushed) continue;
            int u = f.u;
            int pe = f.pe;
            stk.pop_back();
            if (!stk.empty()) {
                int pu = stk.back().u;
                low[pu] = min(low[pu], low[u]);
                if (pe != -1 && !skip[pe] && low[u] > disc[pu]) {
                    is_bridge[pe] = true;
                }
            }
        }
    }

    for (int i = 0; i < m; i++) if (skip[i]) is_bridge[i] = false;
    int count = 0;
    for (int i = 0; i < m; i++) if (is_bridge[i]) count++;
    return count;
}

// =====================================================================
// Main
// =====================================================================
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // 先测mediumDG.txt
    {
        string fn = "mediumDG.txt";
        ifstream fin(fn);
        int n2, m2;
        fin >> n2 >> m2;
        vector<int> eu2(m2), ev2(m2);
        vector<vector<pair<int,int>>> adj2(n2);
        for (int i = 0; i < m2; i++) {
            int u, v; fin >> u >> v;
            eu2[i] = u; ev2[i] = v;
            if (u != v) { adj2[u].push_back({v, i}); adj2[v].push_back({u, i}); }
        }
        fin.close();
        vector<bool> sk2(m2, false);
        for (int i = 0; i < m2; i++) if (eu2[i] == ev2[i]) sk2[i] = true;
        map<pair<int,int>, int> ef2;
        for (int i = 0; i < m2; i++) {
            if (sk2[i]) continue;
            int u = min(eu2[i], ev2[i]), v = max(eu2[i], ev2[i]);
            auto key = make_pair(u, v);
            if (ef2.count(key)) { sk2[i] = true; sk2[ef2[key]] = true; }
            else ef2[key] = i;
        }
        cout << "=== mediumDG.txt: V=" << n2 << " E=" << m2 << " ===\n";
        int b_uf = run_unionfind(n2, m2, eu2, ev2, adj2, sk2);
        int b_tj = run_tarjan(n2, m2, adj2, sk2);
        cout << "并查集: " << b_uf << "桥, Tarjan: " << b_tj << "桥, 一致: " << (b_uf==b_tj?"PASS":"FAIL") << "\n\n";
    }

    string filename = "largeG.txt";

    // 加载图
    auto t0 = chrono::high_resolution_clock::now();
    ifstream fin(filename);
    int n, m;
    fin >> n >> m;
    vector<int> eu(m), ev(m);
    vector<vector<pair<int,int>>> adj_full(n);
    for (int i = 0; i < m; i++) {
        int u, v;
        fin >> u >> v;
        eu[i] = u; ev[i] = v;
        if (u != v) {
            adj_full[u].push_back({v, i});
            adj_full[v].push_back({u, i});
        }
    }
    fin.close();
    auto t1 = chrono::high_resolution_clock::now();
    cout << filename << ": V=" << n << " E=" << m << "\n";
    cout << "加载: " << fixed << setprecision(2) << chrono::duration<double>(t1-t0).count() << "s\n";

    // 预处理
    vector<bool> skip(m, false);
    for (int i = 0; i < m; i++) {
        if (eu[i] == ev[i]) skip[i] = true;
    }
    map<pair<int,int>, int> edge_first;
    for (int i = 0; i < m; i++) {
        if (skip[i]) continue;
        int u = min(eu[i], ev[i]), v = max(eu[i], ev[i]);
        auto key = make_pair(u, v);
        if (edge_first.count(key)) {
            skip[i] = true;
            skip[edge_first[key]] = true;
        } else {
            edge_first[key] = i;
        }
    }
    auto t2 = chrono::high_resolution_clock::now();
    cout << "预处理: " << chrono::duration<double>(t2-t1).count() << "s\n";

    // 并查集算法 (best of 5)
    cout << "\n[并查集算法]\n";
    double t_uf_best = 1e9;
    int uf_bridges = 0;
    for (int run = 0; run < 5; run++) {
        auto ts = chrono::high_resolution_clock::now();
        uf_bridges = run_unionfind(n, m, eu, ev, adj_full, skip);
        auto te = chrono::high_resolution_clock::now();
        double t = chrono::duration<double>(te-ts).count();
        t_uf_best = min(t_uf_best, t);
        cout << "  run " << run << ": " << fixed << setprecision(4) << t << "s\n";
    }

    // Tarjan算法 (best of 5)
    cout << "\n[Tarjan算法]\n";
    double t_tarjan_best = 1e9;
    int tarjan_bridges = 0;
    for (int run = 0; run < 5; run++) {
        auto ts = chrono::high_resolution_clock::now();
        tarjan_bridges = run_tarjan(n, m, adj_full, skip);
        auto te = chrono::high_resolution_clock::now();
        double t = chrono::duration<double>(te-ts).count();
        t_tarjan_best = min(t_tarjan_best, t);
        cout << "  run " << run << ": " << fixed << setprecision(4) << t << "s\n";
    }

    // 结果
    cout << "\n=== 最终结果 ===\n";
    cout << "并查集:  " << fixed << setprecision(4) << t_uf_best << "s, " << uf_bridges << "桥\n";
    cout << "Tarjan:  " << t_tarjan_best << "s, " << tarjan_bridges << "桥\n";
    cout << "UF/Tarjan: " << setprecision(2) << t_uf_best / t_tarjan_best << "x\n";
    cout << "结果一致: " << (uf_bridges == tarjan_bridges ? "PASS" : "FAIL") << "\n";

    return 0;
}
