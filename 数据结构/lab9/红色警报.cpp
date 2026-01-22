#include <bits/stdc++.h>
using namespace std;
typedef long long LL;
typedef std::pair<int, int> PII;
typedef std::pair<LL, LL> PLL;
const int N = 1010;

int t, n, m, k;
vector<int> g[N]; 
bool st[N], lost[N];

void dfs(int u) {
    st[u] = true;
    for (auto &v : g[u]) {
        if (!st[v] && !lost[v]) {
            dfs(v);
        }
    }
}

int components() {
    memset(st, false, sizeof(st));
    int res =  0;
    for (int i = 0; i < n; i++) {
        if (!st[i] && !lost[i]) {
            res++;
            dfs(i);
        }
    }
    return res;
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::cin >> n >> m;
    for (int i = 0; i < m; i++) {
        int a, b;
        std::cin >> a >> b;
        g[a].push_back(b);
        g[b].push_back(a);
    }

    std::cin >> k;
    int currentCount = components();

    for (int i = 0; i < k; i++) {
        int c;
        std::cin >> c;
        lost[c] = true;
        int newCount = components();

        if (newCount > currentCount) {
            cout << "Red Alert: City " << c << " is lost!" << endl;
        } else {
            cout << "City " << c << " is lost." << endl;
        }

        currentCount = newCount;
        if (i == n - 1) {
            cout << "Game Over." << endl;
        }
    }


  
    return 0;
}
