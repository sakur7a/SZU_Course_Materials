#include <bits/stdc++.h>
using namespace std;
typedef long long LL;
typedef std::pair<int, int> PII;
typedef std::pair<LL, LL> PLL;
const int N = 100010;

int t, n, m;

void dfs(string s, int &idx, int depth, vector<int> &d) {
    if (idx >= s.size() || s[idx] == '0') {
        idx++;
        return;
    }

    char c = s[idx++];
    if (isupper(c)) {
        d.push_back(depth);
    }
    dfs(s, idx, depth + 1, d);
    dfs(s, idx, depth + 1, d);
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::cin >> t;
    while (t--) {
        string pre;
        std::cin >> pre >> n;

        vector<int> w(n), d;
        for (int i = 0; i < n; i++) {
            std::cin >> w[i];
        }
        int idx = 0;
        dfs(pre, idx, 0, d);

        int ans = 0;
        for (int i = 0; i < n; i++) {
            ans += w[i] * d[i];
        }

        std::cout << ans << "\n";
    }
  
    return 0;
}


