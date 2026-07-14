#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// Dinic (Optimized)
struct DinicOpt {
    struct Edge { int to, rev; ll cap, flow; };
    int n; vector<vector<Edge>> G; vector<int> lv, it;
    DinicOpt(int n): n(n), G(n), lv(n), it(n) {}
    void add(int u,int v,ll c){
        G[u].push_back({v,(int)G[v].size(),c,0});
        G[v].push_back({u,(int)G[u].size()-1,0,0});
    }
    bool bfs(int s,int t){
        fill(lv.begin(),lv.end(),-1); lv[s]=0;
        queue<int>q; q.push(s);
        while(!q.empty()){int u=q.front();q.pop();
            for(auto&e:G[u]) if(e.cap-e.flow>0&&lv[e.to]<0){
                lv[e.to]=lv[u]+1; if(e.to==t)return true; q.push(e.to);}
        } return lv[t]>=0;
    }
    ll dfs(int u,int t,ll f){
        if(u==t)return f;
        for(int&i=it[u];i<(int)G[u].size();i++){
            Edge&e=G[u][i];
            if(e.cap-e.flow>0&&lv[u]<lv[e.to]){
                ll d=dfs(e.to,t,min(f,e.cap-e.flow));
                if(d>0){e.flow+=d;G[e.to][e.rev].flow-=d;return d;}
            }
        } return 0;
    }
    ll maxflow(int s,int t){
        ll tot=0;
        while(bfs(s,t)){fill(it.begin(),it.end(),0);
            while(1){ll f=dfs(s,t,1e18);if(!f)break;tot+=f;}
        } return tot;
    }
};

// Push-Relabel (HL + Gap)
struct PRHL {
    struct Edge { int to, rev; ll cap, flow; };
    int n; vector<vector<Edge>> G;
    PRHL(int n): n(n), G(n) {}
    void add(int u,int v,ll c){
        G[u].push_back({v,(int)G[v].size(),c,0});
        G[v].push_back({u,(int)G[u].size()-1,0,0});
    }
    ll maxflow(int s,int t){
        vector<int> ht(n,0), gap(2*n+2,0);
        vector<ll> exc(n,0);
        vector<vector<int>> bk(2*n+2);
        vector<char> inb(n,0);
        int mh=0;
        auto act=[&](int v){
            if(v!=s&&v!=t&&exc[v]>0&&!inb[v]){
                int h=ht[v]; bk[h].push_back(v); inb[v]=1;
                if(h>mh) mh=h;
            }
        };
        ht[s]=n; gap[n]=1; gap[0]=n-1;
        for(auto&e:G[s]) if(e.cap>0){
            ll a=e.cap; e.flow+=a; G[e.to][e.rev].flow-=a;
            exc[e.to]+=a; exc[s]-=a; act(e.to);
        }
        while(mh>=0){
            while(mh>=0&&bk[mh].empty()) mh--;
            if(mh<0) break;
            int u=bk[mh].back(); bk[mh].pop_back(); inb[u]=0;
            while(exc[u]>0){
                bool pushed=false;
                for(auto&e:G[u])
                    if(e.cap-e.flow>0&&ht[u]==ht[e.to]+1){
                        ll a=min(exc[u],e.cap-e.flow);
                        e.flow+=a; G[e.to][e.rev].flow-=a;
                        exc[u]-=a; exc[e.to]+=a; act(e.to);
                        if(exc[u]==0){pushed=true;break;} pushed=true;
                    }
                if(exc[u]==0) break;
                if(!pushed){
                    int oh=ht[u], mh2=2*n;
                    for(auto&e:G[u]) if(e.cap-e.flow>0&&ht[e.to]<mh2) mh2=ht[e.to];
                    gap[oh]--;
                    if(gap[oh]==0&&oh<n){
                        for(int v=0;v<n;v++) if(oh<ht[v]&&ht[v]<n){
                            gap[ht[v]]--; ht[v]=n+1; gap[n+1]++; inb[v]=0;
                        }
                        ht[u]=n+1; gap[n+1]++;
                    } else { ht[u]=mh2+1; gap[ht[u]]++; }
                    act(u);
                }
            }
        }
        return exc[t];
    }
};

// Push-Relabel (FIFO + Gap)
struct PRFIFO {
    struct Edge { int to, rev; ll cap, flow; };
    int n; vector<vector<Edge>> G;
    PRFIFO(int n): n(n), G(n) {}
    void add(int u,int v,ll c){
        G[u].push_back({v,(int)G[v].size(),c,0});
        G[v].push_back({u,(int)G[u].size()-1,0,0});
    }
    ll maxflow(int s,int t){
        vector<int> ht(n,0), gap(2*n+2,0);
        vector<ll> exc(n,0);
        queue<int> q;
        vector<char> inq(n,0);
        auto enqueue=[&](int v){
            if(v!=s&&v!=t&&exc[v]>0&&!inq[v]){q.push(v);inq[v]=1;}
        };
        ht[s]=n; gap[n]=1; gap[0]=n-1;
        for(auto&e:G[s]) if(e.cap>0){
            ll a=e.cap; e.flow+=a; G[e.to][e.rev].flow-=a;
            exc[e.to]+=a; exc[s]-=a; enqueue(e.to);
        }
        while(!q.empty()){
            int u=q.front(); q.pop(); inq[u]=0;
            while(exc[u]>0){
                bool pushed=false;
                for(auto&e:G[u])
                    if(e.cap-e.flow>0&&ht[u]==ht[e.to]+1){
                        ll a=min(exc[u],e.cap-e.flow);
                        e.flow+=a; G[e.to][e.rev].flow-=a;
                        exc[u]-=a; exc[e.to]+=a; enqueue(e.to);
                        if(exc[u]==0){pushed=true;break;} pushed=true;
                    }
                if(exc[u]==0) break;
                if(!pushed){
                    int oh=ht[u], mh=2*n;
                    for(auto&e:G[u]) if(e.cap-e.flow>0&&ht[e.to]<mh) mh=ht[e.to];
                    gap[oh]--;
                    if(gap[oh]==0&&oh<n){
                        for(int v=0;v<n;v++) if(oh<ht[v]&&ht[v]<n){
                            gap[ht[v]]--; ht[v]=n+1; gap[n+1]++; inq[v]=0;
                        }
                        ht[u]=n+1; gap[n+1]++;
                    } else { ht[u]=mh+1; gap[ht[u]]++; }
                    enqueue(u);
                }
            }
        }
        return exc[t];
    }
};

// Ford-Fulkerson (DFS)
struct FF {
    struct Edge { int to, rev; ll cap, flow; };
    int n; vector<vector<Edge>> G;
    FF(int n): n(n), G(n) {}
    void add(int u,int v,ll c){
        G[u].push_back({v,(int)G[v].size(),c,0});
        G[v].push_back({u,(int)G[u].size()-1,0,0});
    }
    ll dfs(int u,int t,ll f,vector<char>&vis){
        if(u==t) return f;
        vis[u]=1;
        for(auto&e:G[u]) if(!vis[e.to]&&e.cap-e.flow>0){
            ll d=dfs(e.to,t,min(f,e.cap-e.flow),vis);
            if(d>0){e.flow+=d;G[e.to][e.rev].flow-=d;return d;}
        }
        return 0;
    }
    ll maxflow(int s,int t){
        ll tot=0;
        while(1){
            vector<char> vis(n,0);
            ll f=dfs(s,t,1e18,vis);
            if(!f) break;
            tot+=f;
        }
        return tot;
    }
};

// Edmonds-Karp
struct EK {
    struct Edge { int to, rev; ll cap, flow; };
    int n; vector<vector<Edge>> G;
    EK(int n): n(n), G(n) {}
    void add(int u,int v,ll c){
        G[u].push_back({v,(int)G[v].size(),c,0});
        G[v].push_back({u,(int)G[u].size()-1,0,0});
    }
    ll maxflow(int s,int t){
        ll tot=0; vector<Edge*> par(n);
        while(1){
            fill(par.begin(),par.end(),nullptr);
            queue<int>q; q.push(s);
            while(!q.empty()&&!par[t]){int u=q.front();q.pop();
                for(auto&e:G[u]) if(!par[e.to]&&e.cap-e.flow>0){
                    par[e.to]=&e; q.push(e.to);}
            }
            if(!par[t]) break;
            ll pf=1e18;
            for(int v=t;v!=s;) { pf=min(pf,par[v]->cap-par[v]->flow); v=G[v][par[v]->rev].to; }
            for(int v=t;v!=s;) { par[v]->flow+=pf; G[v][par[v]->rev].flow-=pf; v=G[v][par[v]->rev].to; }
            tot+=pf;
        }
        return tot;
    }
};

// ═══════════════════════════════════════════════════════════════
// 图生成器 — 五种测试图类型
// ═══════════════════════════════════════════════════════════════

namespace GraphGen {
    const int INF_CAP = 1000000000;  // 10亿, 在int范围内且足够充当"无限"

    // ── 1. 层次DAG ─────────────────────────────────
    // 节点按索引分配到 L≈√n 层，边仅从低层到高层（最多跨2层）
    // 作用：测大规模稠密随机网络吞吐，Dinic层次图优势场景
    vector<tuple<int,int,int>> shallow_dag(int n, mt19937& rng) {
        int L = max(3, min(30, (int)sqrt(n)));
        vector<vector<int>> layer(L);
        layer[0].push_back(0);               // 源点在第0层
        layer[L-1].push_back(n-1);           // 汇点在最后一层
        for (int i = 1; i < n-1; i++) {
            int l = 1 + (i-1) * (L-2) / (n-2);
            layer[l].push_back(i);
        }
        uniform_real_distribution<double> prob(0,1);
        uniform_int_distribution<int> cap(1,100);
        vector<tuple<int,int,int>> edges;
        for (int l = 0; l < L-1; l++) {
            for (int u : layer[l]) {
                // 连到下一层 (密度 0.25)
                for (int v : layer[l+1])
                    if (prob(rng) < 0.25)
                        edges.push_back({u, v, cap(rng)});
                // 跨层连接 l → l+2 (密度 0.08)
                if (l+2 < L)
                    for (int v : layer[l+2])
                        if (prob(rng) < 0.08)
                            edges.push_back({u, v, cap(rng)});
            }
        }
        // 骨干链保证连通
        for (int l = 0; l < L-1; l++)
            if (!layer[l].empty() && !layer[l+1].empty())
                edges.push_back({layer[l][0], layer[l+1][0], cap(rng)});
        return edges;
    }

    // ── 2. 多层网络（棒球流 / 二分图结构）──────────────
    // 结构: S(0) → 比赛节点 → 球队节点 → T(n-1)
    // 每个"比赛"连接恰好2个"球队"，模拟棒球淘汰问题的流网络拓扑
    // 作用：更接近实际应用的流网络结构
    vector<tuple<int,int,int>> multi_layer(int n, mt19937& rng) {
        int G = max(10, (n - 2) * 2 / 5);  // ~40% 比赛节点
        int T = n - 2 - G;                  // ~60% 球队节点
        if (T < 2) { G = n - 4; T = 2; }

        uniform_int_distribution<int> gcap(1, 30);
        uniform_int_distribution<int> tcap(1, 80);
        uniform_int_distribution<int> tsel(1, T);
        vector<tuple<int,int,int>> edges;

        // S → 比赛节点 (idx 1..G)
        for (int g = 1; g <= G; g++)
            edges.push_back({0, g, gcap(rng)});

        // 比赛节点 → 2个球队节点 (idx G+1 .. G+T)
        for (int g = 1; g <= G; g++) {
            int t1 = G + tsel(rng);
            int t2 = G + tsel(rng);
            while (t2 == t1) t2 = G + tsel(rng);
            edges.push_back({g, t1, INF_CAP});
            edges.push_back({g, t2, INF_CAP});
        }

        // 球队节点 → T(n-1)
        for (int t = G+1; t <= G+T; t++)
            edges.push_back({t, n-1, tcap(rng)});

        return edges;
    }

    // ── 3. 带窄割瓶颈的网络 ────────────────────────────
    // 两个内部稠密的簇，仅靠k条小容量边连接 → 明确的最小割瓶颈
    // 大n时直接随机生成边（避免O(n²)枚举），每簇约 n*5 条边
    // 作用：测算法处理瓶颈能力——PR的gap启发式在此优势明显
    vector<tuple<int,int,int>> narrow_cut(int n, int k, mt19937& rng) {
        int mid = n / 2;
        uniform_int_distribution<int> cap(1, 20);
        vector<tuple<int,int,int>> edges;

        int target_per_cluster = n * 5;
        auto gen_cluster = [&](int L, int R) {
            if (R - L + 1 <= 2) return;
            uniform_int_distribution<int> u_gen(L, R-1);
            for (int t = 0; t < target_per_cluster; t++) {
                int u = u_gen(rng);
                int v = uniform_int_distribution<int>(u+1, R)(rng);
                edges.push_back({u, v, cap(rng)});
            }
        };
        gen_cluster(0, mid-1);
        gen_cluster(mid, n-1);

        // 骨干链保证s↝t连通
        for (int i = 0; i < mid-1; i++)
            edges.push_back({i, i+1, cap(rng)});
        for (int i = mid; i < n-1; i++)
            edges.push_back({i, i+1, cap(rng)});

        // 瓶颈边: k条 簇0→簇1，小容量
        uniform_int_distribution<int> bl(0, mid-1);
        uniform_int_distribution<int> br(mid, n-1);
        uniform_int_distribution<int> bcap(1, 5);
        for (int i = 0; i < k; i++)
            edges.push_back({bl(rng), br(rng), bcap(rng)});

        return edges;
    }

    // ── 4. 长链网络 ────────────────────────────────────
    // 节点排成链 0→1→2→…→n-1，加少量前向跳边
    // 作用：测长增广路径场景（FF/E-K退化为O(F·E)，Dinic层次数=O(n)）
    vector<tuple<int,int,int>> long_chain(int n, mt19937& rng) {
        uniform_int_distribution<int> cap(1, 200);
        uniform_real_distribution<double> prob(0,1);
        vector<tuple<int,int,int>> edges;

        // 主链
        for (int i = 0; i < n-1; i++)
            edges.push_back({i, i+1, cap(rng)});

        // 前向跳边 i→i+skip（低密度）
        for (int i = 0; i < n-3; i++)
            for (int j = i+2; j < min(n, i+12); j++)
                if (prob(rng) < 0.03)
                    edges.push_back({i, j, cap(rng)});

        return edges;
    }

    // ── 5. CLRS经典图 ──────────────────────────────────
    // 固定6节点网络，已知最大流=23，含反向边(capacity>0的双向边)
    // 作用：测正确性和反向边撤销（所有算法必须返回23）
    vector<tuple<int,int,int>> clrs_classic() {
        return {
            {0,1,16}, {0,2,13}, {1,2,10}, {1,3,12},
            {2,1,4},  {2,4,14}, {3,2,9},  {3,5,20},
            {4,3,7},  {4,5,4}
        };
    }
}

// ═══════════════════════════════════════════════════════════════

int main(){
    ios::sync_with_stdio(false); cin.tie(nullptr);
    const double TMO = 60.0;

    // ═══════════════════════════════════════════════════════════
    // 阶段1: FF + E-K 小规模实测 + 外推
    // ═══════════════════════════════════════════════════════════
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  阶段1: FF / E-K 实测 + 外推 (随机图)                   ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");
    vector<pair<int,double>> ff_t, ek_t;
    for(int n : {500, 1000, 2000, 3000, 5000, 10000}){
        mt19937 rng(42);
        uniform_real_distribution<double> prob(0,1);
        uniform_int_distribution<int> cap(1,20);
        vector<tuple<int,int,int>> edges;
        for(int i=0;i<n;i++) for(int j=i+1;j<n;j++) if(prob(rng)<0.02)
            edges.push_back({i,j,cap(rng)});
        { FF G(n); for(auto&[u,v,c]:edges) G.add(u,v,c);
          auto t0=chrono::high_resolution_clock::now();
          G.maxflow(0,n-1);
          double sec=chrono::duration<double>(chrono::high_resolution_clock::now()-t0).count();
          ff_t.push_back({n,sec}); printf("  FF  n=%5d: %.3fs\n",n,sec); }
        { EK G(n); for(auto&[u,v,c]:edges) G.add(u,v,c);
          auto t0=chrono::high_resolution_clock::now();
          G.maxflow(0,n-1);
          double sec=chrono::duration<double>(chrono::high_resolution_clock::now()-t0).count();
          ek_t.push_back({n,sec}); printf("  E-K n=%5d: %.3fs\n",n,sec); }
    }
    auto fit=[](vector<pair<int,double>>&pts)->pair<double,double>{
        double sx=0,sy=0,sxx=0,sxy=0; int m=pts.size();
        for(auto&[n,t]:pts){double x=log(n),y=log(t);sx+=x;sy+=y;sxx+=x*x;sxy+=x*y;}
        double b=(m*sxy-sx*sy)/(m*sxx-sx*sx);
        double a=exp((sy-b*sx)/m);
        return {a,b};
    };
    auto [ff_a,ff_b] = fit(ff_t);
    auto [ek_a,ek_b] = fit(ek_t);
    printf("  FF  外推: t = %.3e * n^%.2f\n",ff_a,ff_b);
    printf("  E-K 外推: t = %.3e * n^%.2f\n",ek_a,ek_b);

    // ═══════════════════════════════════════════════════════════
    // 阶段2: 随机图大规模基准测试 (原测试保留)
    // ═══════════════════════════════════════════════════════════
    printf("\n╔══════════════════════════════════════════════════════════╗\n");
    printf("║  阶段2: 随机图大规模基准 (边密度 2%%)                    ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    auto fmt_time = [](double s)->string{
        if(s<0) return ">60";
        if(s<1e-3) {char b[32];sprintf(b,"%.1e",s);return b;}
        if(s<1) {char b[32];sprintf(b,"%.3f",s);return b;}
        if(s<10) {char b[32];sprintf(b,"%.2f",s);return b;}
        if(s<100) {char b[32];sprintf(b,"%.1f",s);return b;}
        if(s<3600) {char b[32];sprintf(b,"%.0f",s);return b;}
        {char b[32];sprintf(b,"%.1e",s);return b;}
    };

    vector<pair<int,double>> configs = {
        {1000, 0.02}, {5000, 0.02}, {10000, 0.02},
        {50000, 0.02}, {100000, 0.02}
    };

    printf("%8s %12s | %8s %8s %8s %8s %8s\n",
           "节点","边数","FF","E-K","Dinic","PR(FIFO)","PR(HL)");
    printf("%s\n", string(78,'-').c_str());

    // 构建实测值查找表 (阶段1已测 n=500..10000)
    unordered_map<int,double> ff_real, ek_real;
    for(auto&[n,t] : ff_t) ff_real[n] = t;
    for(auto&[n,t] : ek_t) ek_real[n] = t;

    for(auto&[n,p] : configs){
        mt19937 rng(42);
        uniform_real_distribution<double> prob(0,1);
        uniform_int_distribution<int> cap(1,20);
        vector<tuple<int,int,int>> edges;
        for(int i=0;i<n;i++) for(int j=i+1;j<n;j++) if(prob(rng)<p)
            edges.push_back({i,j,cap(rng)});

        // FF/EK: n≤10000用实测，更大用外推
        bool ff_is_real = ff_real.count(n);
        bool ek_is_real = ek_real.count(n);
        double ff_val = ff_is_real ? ff_real[n] : ff_a*pow((double)n,ff_b);
        double ek_val = ek_is_real ? ek_real[n] : ek_a*pow((double)n,ek_b);
        string ff_s = fmt_time(ff_val) + (ff_is_real ? " " : "*");
        string ek_s = fmt_time(ek_val) + (ek_is_real ? " " : "*");
        double din_sec=-1, prf_sec=-1, prh_sec=-1;

        {
            DinicOpt G(n); for(auto&[u,v,c]:edges) G.add(u,v,c);
            auto t0=chrono::high_resolution_clock::now();
            G.maxflow(0,n-1);
            din_sec=chrono::duration<double>(chrono::high_resolution_clock::now()-t0).count();
            if(din_sec>TMO) din_sec=-1;
        }
        {
            PRFIFO G(n); for(auto&[u,v,c]:edges) G.add(u,v,c);
            auto t0=chrono::high_resolution_clock::now();
            G.maxflow(0,n-1);
            prf_sec=chrono::duration<double>(chrono::high_resolution_clock::now()-t0).count();
            if(prf_sec>TMO) prf_sec=-1;
        }
        {
            PRHL G(n); for(auto&[u,v,c]:edges) G.add(u,v,c);
            auto t0=chrono::high_resolution_clock::now();
            G.maxflow(0,n-1);
            prh_sec=chrono::duration<double>(chrono::high_resolution_clock::now()-t0).count();
            if(prh_sec>TMO) prh_sec=-1;
        }

        printf("%8d %12d | %8s %8s %8s %8s %8s\n",
               n,(int)edges.size(),
               ff_s.c_str(),
               ek_s.c_str(),
               din_sec<0?">60":fmt_time(din_sec).c_str(),
               prf_sec<0?">60":fmt_time(prf_sec).c_str(),
               prh_sec<0?">60":fmt_time(prh_sec).c_str());
    }
    printf("\n  FF/E-K: n≤10000为实测, n>10000带*为外推 (t=a*n^b)\n");
    printf("  超时阈值: %.0fs\n",TMO);

    // ═══════════════════════════════════════════════════════════
    // 阶段3: 多类型图基准测试 (五种图 × 多种规模)
    // ═══════════════════════════════════════════════════════════
    printf("\n\n╔══════════════════════════════════════════════════════════╗\n");
    printf("║  阶段3: 多类型图基准测试                                ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");

    struct GraphConfig {
        string name;
        string desc;
        function<vector<tuple<int,int,int>>(int,mt19937&)> gen;
        vector<int> sizes;
    };

    vector<GraphConfig> graph_types = {
        {"层次DAG", "L≈√n层, 高密度跨层边, 测吞吐",
         [](int n, mt19937& rng){ return GraphGen::shallow_dag(n, rng); },
         {10000, 50000, 100000}},
        {"多层网络", "S→比赛→球队→T, 二分图结构",
         [](int n, mt19937& rng){ return GraphGen::multi_layer(n, rng); },
         {10000, 50000, 100000}},
        {"窄割瓶颈", "双簇+n*5条边+少量桥边, 显式瓶颈",
         [](int n, mt19937& rng){ return GraphGen::narrow_cut(n, 5, rng); },
         {10000, 50000, 100000}},
        {"长链网络", "主链+少量跳边, 深层次图",
         [](int n, mt19937& rng){ return GraphGen::long_chain(n, rng); },
         {10000, 50000, 100000}},
    };

    // 先验证CLRS经典图正确性
    printf("\n── CLRS经典图正确性验证 ──────────────────────────────\n");
    {
        auto edges = GraphGen::clrs_classic();
        ll ref = 23;
        printf("  参考最大流 = %lld\n", ref);

        { DinicOpt G(6); for(auto&[u,v,c]:edges) G.add(u,v,c);
          ll f = G.maxflow(0,5);
          printf("  Dinic(Opt)      : %5lld  %s\n", f, f==ref?"OK":"WRONG"); }
        { PRFIFO G(6); for(auto&[u,v,c]:edges) G.add(u,v,c);
          ll f = G.maxflow(0,5);
          printf("  PR(FIFO+Gap)    : %5lld  %s\n", f, f==ref?"OK":"WRONG"); }
        { PRHL G(6); for(auto&[u,v,c]:edges) G.add(u,v,c);
          ll f = G.maxflow(0,5);
          printf("  PR(HL+Gap)      : %5lld  %s\n", f, f==ref?"OK":"WRONG"); }
    }

    // 各图类型逐规模测试
    for (auto& gt : graph_types) {
        printf("\n── %s ─────────────────────────────────────────────\n", gt.name.c_str());
        printf("  说明: %s\n", gt.desc.c_str());
        printf("  %8s %12s | %8s %8s %8s | %8s\n",
               "规模","边数","Dinic","PR(FIFO)","PR(HL)","备注");

        for (int n : gt.sizes) {
            mt19937 rng(42+n);  // 不同图类型用不同种子但可复现
            auto edges = gt.gen(n, rng);
            int m = (int)edges.size();

            double d_sec=-1, f_sec=-1, h_sec=-1;
            ll f_dinic=0, f_prf=0, f_prh=0;

            {
                DinicOpt G(n); for(auto&[u,v,c]:edges) G.add(u,v,c);
                auto t0=chrono::high_resolution_clock::now();
                f_dinic = G.maxflow(0,n-1);
                d_sec=chrono::duration<double>(chrono::high_resolution_clock::now()-t0).count();
                if(d_sec>TMO) d_sec=-1;
            }
            {
                PRFIFO G(n); for(auto&[u,v,c]:edges) G.add(u,v,c);
                auto t0=chrono::high_resolution_clock::now();
                f_prf = G.maxflow(0,n-1);
                f_sec=chrono::duration<double>(chrono::high_resolution_clock::now()-t0).count();
                if(f_sec>TMO) f_sec=-1;
            }
            {
                PRHL G(n); for(auto&[u,v,c]:edges) G.add(u,v,c);
                auto t0=chrono::high_resolution_clock::now();
                f_prh = G.maxflow(0,n-1);
                h_sec=chrono::duration<double>(chrono::high_resolution_clock::now()-t0).count();
                if(h_sec>TMO) h_sec=-1;
            }

            // 验证一致性
            string note = "";
            if (f_dinic != f_prf || f_prf != f_prh)
                note = "⚠流量不一致";
            else if (d_sec>0 && f_sec>0 && h_sec>0) {
                double fastest = min({d_sec, f_sec, h_sec});
                if (fastest == h_sec && h_sec > 0.001) note = "HL最快";
                else if (fastest == d_sec && d_sec > 0.001) note = "Dinic最快";
                else if (fastest == f_sec && f_sec > 0.001) note = "FIFO最快";
            }

            printf("  %8d %12d | %8s %8s %8s | %s\n",
                   n, m,
                   d_sec<0?">60":fmt_time(d_sec).c_str(),
                   f_sec<0?">60":fmt_time(f_sec).c_str(),
                   h_sec<0?">60":fmt_time(h_sec).c_str(),
                   note.c_str());
        }
    }

    // 总结
    printf("\n\n╔══════════════════════════════════════════════════════════╗\n");
    printf("║  测试总结                                               ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");
    printf("  图类型          主要测试目标\n");
    printf("  ──────────────  ──────────────────────────────────\n");
    printf("  随机图          通用吞吐量 (baseline)\n");
    printf("  层次DAG     稠密层次图, Dinic优势场景\n");
    printf("  多层网络         棒球流/二分图结构, 应用场景覆盖\n");
    printf("  窄割瓶颈         显式瓶颈, PR gap启发式优势\n");
    printf("  长链网络         深层次图, 增广路径长度影响\n");
    printf("  CLRS经典图       正确性验证, 反向边撤销\n");

    return 0;
}
