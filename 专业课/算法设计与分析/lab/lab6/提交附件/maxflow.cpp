/*
 * 最大流算法 C++ 实现
 * 包含：Edmonds-Karp、Dinic(Optimized)、Push-Relabel(HL+Gap)
 * 以及棒球淘汰问题求解器
 */

#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
const ll INF = 1e18;

// ═══════════════════════════════════════════════════════════════
// 通用边结构
// ═══════════════════════════════════════════════════════════════

struct Edge {
    int to, rev;
    ll cap, flow;
};

// ═══════════════════════════════════════════════════════════════
// Edmonds-Karp
// ═══════════════════════════════════════════════════════════════

struct EdmondsKarp {
    int n;
    vector<vector<Edge>> graph;

    EdmondsKarp(int n) : n(n), graph(n) {}

    void add_edge(int u, int v, ll cap) {
        graph[u].push_back({v, (int)graph[v].size(), cap, 0});
        graph[v].push_back({u, (int)graph[u].size() - 1, 0, 0});
    }

    ll max_flow(int s, int t) {
        ll total = 0;
        vector<Edge*> parent(n);
        while (true) {
            fill(parent.begin(), parent.end(), nullptr);
            queue<int> q;
            q.push(s);
            while (!q.empty() && !parent[t]) {
                int u = q.front(); q.pop();
                for (auto &e : graph[u]) {
                    if (!parent[e.to] && e.cap - e.flow > 0) {
                        parent[e.to] = &e;
                        q.push(e.to);
                    }
                }
            }
            if (!parent[t]) break;
            ll path_flow = INF;
            for (int v = t; v != s; ) {
                path_flow = min(path_flow, parent[v]->cap - parent[v]->flow);
                v = graph[v][parent[v]->rev].to;
            }
            for (int v = t; v != s; ) {
                parent[v]->flow += path_flow;
                graph[v][parent[v]->rev].flow -= path_flow;
                v = graph[v][parent[v]->rev].to;
            }
            total += path_flow;
        }
        return total;
    }
};

// ═══════════════════════════════════════════════════════════════
// Dinic (Optimized)：当前弧优化 + BFS提前终止
// ═══════════════════════════════════════════════════════════════

struct DinicOpt {
    int n;
    vector<vector<Edge>> graph;
    vector<int> level, iter;

    DinicOpt(int n) : n(n), graph(n), level(n), iter(n) {}

    void add_edge(int u, int v, ll cap) {
        graph[u].push_back({v, (int)graph[v].size(), cap, 0});
        graph[v].push_back({u, (int)graph[u].size() - 1, 0, 0});
    }

    bool bfs(int s, int t) {
        fill(level.begin(), level.end(), -1);
        level[s] = 0;
        queue<int> q;
        q.push(s);
        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (auto &e : graph[u]) {
                if (e.cap - e.flow > 0 && level[e.to] < 0) {
                    level[e.to] = level[u] + 1;
                    if (e.to == t) return true;
                    q.push(e.to);
                }
            }
        }
        return level[t] >= 0;
    }

    ll dfs(int u, int t, ll f) {
        if (u == t) return f;
        for (int &i = iter[u]; i < (int)graph[u].size(); i++) {
            Edge &e = graph[u][i];
            if (e.cap - e.flow > 0 && level[u] < level[e.to]) {
                ll d = dfs(e.to, t, min(f, e.cap - e.flow));
                if (d > 0) {
                    e.flow += d;
                    graph[e.to][e.rev].flow -= d;
                    return d;
                }
            }
        }
        return 0;
    }

    ll max_flow(int s, int t) {
        ll total = 0;
        while (bfs(s, t)) {
            fill(iter.begin(), iter.end(), 0);
            while (true) {
                ll f = dfs(s, t, INF);
                if (f == 0) break;
                total += f;
            }
        }
        return total;
    }
};

// ═══════════════════════════════════════════════════════════════
// Push-Relabel (Highest-Label + Gap启发式)
// 理论复杂度 O(V²√E)，实践中最快
// ═══════════════════════════════════════════════════════════════

struct PushRelabelHLGap {
    int n;
    vector<vector<Edge>> graph;

    PushRelabelHLGap(int n) : n(n), graph(n) {}

    void add_edge(int u, int v, ll cap) {
        graph[u].push_back({v, (int)graph[v].size(), cap, 0});
        graph[v].push_back({u, (int)graph[u].size() - 1, 0, 0});
    }

    ll max_flow(int s, int t) {
        vector<int> height(n, 0), gap(2 * n + 2, 0);
        vector<ll> excess(n, 0);
        vector<vector<int>> bucket(2 * n + 2);
        vector<char> in_bucket(n, 0);
        int max_h = 0;

        auto activate = [&](int v) {
            if (v != s && v != t && excess[v] > 0 && !in_bucket[v]) {
                int h = height[v];
                bucket[h].push_back(v);
                in_bucket[v] = 1;
                if (h > max_h) max_h = h;
            }
        };

        // 初始化：源点推流
        height[s] = n;
        gap[n] = 1;
        gap[0] = n - 1;
        for (auto &e : graph[s]) {
            if (e.cap > 0) {
                ll amt = e.cap;
                e.flow += amt;
                graph[e.to][e.rev].flow -= amt;
                excess[e.to] += amt;
                excess[s] -= amt;
                activate(e.to);
            }
        }

        // 主循环：选择高度最高的活跃节点
        while (max_h >= 0) {
            while (max_h >= 0 && bucket[max_h].empty()) max_h--;
            if (max_h < 0) break;

            int u = bucket[max_h].back();
            bucket[max_h].pop_back();
            in_bucket[u] = 0;

            // Discharge u
            while (excess[u] > 0) {
                bool pushed = false;
                for (auto &e : graph[u]) {
                    if (e.cap - e.flow > 0 && height[u] == height[e.to] + 1) {
                        ll amt = min(excess[u], e.cap - e.flow);
                        e.flow += amt;
                        graph[e.to][e.rev].flow -= amt;
                        excess[u] -= amt;
                        excess[e.to] += amt;
                        activate(e.to);
                        if (excess[u] == 0) { pushed = true; break; }
                        pushed = true;
                    }
                }
                if (excess[u] == 0) break;

                if (!pushed) {
                    // Relabel
                    int old_h = height[u];
                    int min_h = 2 * n;
                    for (auto &e : graph[u]) {
                        if (e.cap - e.flow > 0 && height[e.to] < min_h)
                            min_h = height[e.to];
                    }

                    // Gap启发式
                    gap[old_h]--;
                    if (gap[old_h] == 0 && old_h < n) {
                        for (int v = 0; v < n; v++) {
                            if (old_h < height[v] && height[v] < n) {
                                gap[height[v]]--;
                                height[v] = n + 1;
                                gap[n + 1]++;
                                in_bucket[v] = 0;
                            }
                        }
                        height[u] = n + 1;
                        gap[n + 1]++;
                    } else {
                        height[u] = min_h + 1;
                        gap[height[u]]++;
                    }
                    activate(u);
                }
            }
        }

        return excess[t];
    }
};

// ═══════════════════════════════════════════════════════════════
// 棒球淘汰问题求解器
// ═══════════════════════════════════════════════════════════════

struct BaseballElimination {
    int n;
    vector<string> names;
    vector<int> wins, remaining;
    vector<vector<int>> games;

    BaseballElimination(vector<string> names, vector<int> wins,
                        vector<int> remaining, vector<vector<int>> games)
        : n(names.size()), names(names), wins(wins),
          remaining(remaining), games(games) {}

    // 简单淘汰检测
    pair<bool, vector<string>> is_trivially_eliminated(int x) {
        int max_wins = wins[x] + remaining[x];
        for (int i = 0; i < n; i++) {
            if (i != x && wins[i] > max_wins) {
                return {true, {names[i]}};
            }
        }
        return {false, {}};
    }

    // 用最大流判断是否被淘汰
    pair<bool, vector<string>> is_eliminated(int x) {
        // 先检查简单淘汰
        auto [trivial, proof] = is_trivially_eliminated(x);
        if (trivial) return {true, proof};

        int max_wins = wins[x] + remaining[x];

        // 收集比赛节点（不涉及x的比赛）
        struct GameNode { int i, j, cnt; };
        vector<GameNode> game_nodes;
        for (int i = 0; i < n; i++) {
            if (i == x) continue;
            for (int j = i + 1; j < n; j++) {
                if (j == x) continue;
                if (games[i][j] > 0) game_nodes.push_back({i, j, games[i][j]});
            }
        }

        int m = game_nodes.size();
        if (m == 0) return {false, {}};

        // 构建流网络
        // 节点: 0=源点, 1..m=比赛节点, m+1..m+n-1=球队节点, m+n=汇点
        int s = 0, t = m + n;
        PushRelabelHLGap G(t + 1);

        // 源点到比赛节点
        for (int k = 0; k < m; k++) {
            G.add_edge(s, k + 1, game_nodes[k].cnt);
            // 比赛节点到球队节点（容量INF）
            G.add_edge(k + 1, m + 1 + game_nodes[k].i, INF);
            G.add_edge(k + 1, m + 1 + game_nodes[k].j, INF);
        }

        // 球队节点到汇点
        for (int i = 0; i < n; i++) {
            if (i == x) continue;
            ll cap = max_wins - wins[i];
            if (cap < 0) cap = 0;
            G.add_edge(m + 1 + i, t, cap);
        }

        // 计算最大流
        ll total_games = 0;
        for (auto &g : game_nodes) total_games += g.cnt;

        ll flow = G.max_flow(s, t);

        if (flow == total_games) {
            return {false, {}};
        }

        // 找淘汰证明：BFS找源点可达的球队
        vector<char> vis(t + 1, 0);
        queue<int> q;
        q.push(s);
        vis[s] = 1;
        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (auto &e : G.graph[u]) {
                if (!vis[e.to] && e.cap - e.flow > 0) {
                    vis[e.to] = 1;
                    q.push(e.to);
                }
            }
        }

        vector<string> elim_proof;
        for (int i = 0; i < n; i++) {
            if (i == x) continue;
            if (vis[m + 1 + i]) elim_proof.push_back(names[i]);
        }
        return {true, elim_proof};
    }

    // 求解所有球队
    void solve() {
        cout << string(70, '=') << endl;
        cout << "棒球淘汰问题求解结果" << endl;
        cout << string(70, '=') << endl;

        cout << "\n球队信息:" << endl;
        cout << string(70, '-') << endl;
        printf("%-10s %6s %6s %8s %8s\n", "球队", "已胜", "剩余", "最大可能", "状态");
        cout << string(70, '-') << endl;

        for (int i = 0; i < n; i++) {
            auto [elim, proof] = is_eliminated(i);
            printf("%-10s %6d %6d %8d %8s\n",
                   names[i].c_str(), wins[i], remaining[i],
                   wins[i] + remaining[i], elim ? "淘汰" : "未淘汰");
        }

        cout << "\n淘汰证明:" << endl;
        cout << string(70, '-') << endl;
        for (int i = 0; i < n; i++) {
            auto [elim, proof] = is_eliminated(i);
            if (elim && !proof.empty()) {
                cout << "\n" << names[i] << " 被以下球队淘汰:" << endl;
                for (auto &t : proof) cout << "  - " << t << endl;
            }
        }
    }
};

// ═══════════════════════════════════════════════════════════════
// 基准测试
// ═══════════════════════════════════════════════════════════════

void benchmark_classic() {
    cout << "\n经典6节点网络验证 (最大流=23)" << endl;
    cout << string(50, '-') << endl;

    vector<tuple<int,int,int>> edges = {
        {0,1,16}, {0,2,13}, {1,2,10}, {1,3,12},
        {2,1,4}, {2,4,14}, {3,2,9}, {3,5,20},
        {4,3,7}, {4,5,4}
    };

    auto test = [&](string name, auto &G) {
        for (auto &[u,v,c] : edges) G.add_edge(u,v,c);
        ll flow = G.max_flow(0, 5);
        printf("  %-25s: %lld  %s\n", name.c_str(), flow, flow == 23 ? "OK" : "WRONG");
    };

    { EdmondsKarp G(6); test("Edmonds-Karp", G); }
    { DinicOpt G(6); test("Dinic (Optimized)", G); }
    { PushRelabelHLGap G(6); test("PR (HL+Gap)", G); }
}

void benchmark_large() {
    cout << "\n大规模基准测试 (边密度2%)" << endl;
    cout << string(70, '=') << endl;
    printf("%8s %12s %14s %14s %8s\n", "节点", "边数", "Dinic(Opt)", "PR(HL+Gap)", "加速比");
    cout << string(70, '-') << endl;

    vector<int> sizes = {1000, 5000, 10000, 50000, 100000};

    for (int n : sizes) {
        mt19937 rng(42);
        uniform_real_distribution<double> prob(0.0, 1.0);
        uniform_int_distribution<int> cap(1, 20);

        vector<tuple<int,int,int>> edges;
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                if (prob(rng) < 0.02)
                    edges.push_back({i, j, cap(rng)});

        // Dinic(Opt)
        DinicOpt G1(n);
        for (auto &[u,v,c] : edges) G1.add_edge(u,v,c);
        auto t0 = chrono::high_resolution_clock::now();
        ll flow1 = G1.max_flow(0, n-1);
        auto t1 = chrono::high_resolution_clock::now();
        double ms1 = chrono::duration<double, milli>(t1 - t0).count();

        // PR(HL+Gap)
        PushRelabelHLGap G2(n);
        for (auto &[u,v,c] : edges) G2.add_edge(u,v,c);
        auto t2 = chrono::high_resolution_clock::now();
        ll flow2 = G2.max_flow(0, n-1);
        auto t3 = chrono::high_resolution_clock::now();
        double ms2 = chrono::duration<double, milli>(t3 - t2).count();

        printf("%8d %12d %11.0f ms %11.0f ms %7.1fx\n",
               n, (int)edges.size(), ms1, ms2, ms1 / ms2);

        if (flow1 != flow2) {
            cerr << "  ERROR: flow mismatch! Dinic=" << flow1 << " PR=" << flow2 << endl;
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// 主函数
// ═══════════════════════════════════════════════════════════════

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // ── 1. 经典验证 ──
    benchmark_classic();

    // ── 2. 四支球队淘汰问题 ──
    cout << "\n" << string(70, '=') << endl;
    cout << "四支球队淘汰问题（实验数据）" << endl;

    BaseballElimination solver(
        {"Atlanta", "Philly", "New York", "Montreal"},
        {83, 80, 78, 77},                           // 已胜
        {8, 3, 6, 3},                                // 剩余
        {{0, 1, 6, 1},                                // 对阵矩阵
         {1, 0, 0, 2},
         {6, 0, 0, 0},
         {1, 2, 0, 0}}
    );
    solver.solve();

    // ── 3. 大规模基准测试 ──
    benchmark_large();

    return 0;
}
