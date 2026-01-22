#include<bits/stdc++.h>
typedef long long LL;
typedef std::pair<int, int> PII;
const int N = 100010;

char p[N], s[N];
int n, m, t, pos;
int ne[N];


void solve() {
    std::cin >> s + 1 >> p + 1;

    n = strlen(p + 1);
    m = strlen(s + 1);

    pos = 0;
    ne[0] = -1;

    for (int i = 2, j = 0; i <= n; i++) {
        while (j && p[i] != p[j + 1]) j = ne[j];
        if (p[i] == p[j + 1]) j++;
        ne[i] = j;
    }

    for (int i = 1, j = 0; i <= m; i++) {
        while (j && s[i] != p[j + 1]) j = ne[j];
        if (s[i] == p[j + 1]) j++;
        if (j == n) {
            pos = i - n + 1;
            break;
        }
    }


    for (int i = 0; i < n; i++) {
        std::cout << ne[i] << " ";
    }
    std::cout << "\n" << pos << "\n";

}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::cin >> t;

    while (t--) {
        solve();
    }
  
    return 0;
}
