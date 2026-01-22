#include<bits/stdc++.h>
typedef long long LL;
typedef std::pair<int, int> PII;
const int N = 100010;

char p[N], s[N];
int n, m, t, pos;
int ne[N];


void solve() {
    std::cin >> p + 1;

    n = strlen(p + 1);


    for (int i = 2, j = 0; i <= n; i++) {
        while (j && p[i] != p[j + 1]) j = ne[j];
        if (p[i] == p[j + 1]) j++;
        ne[i] = j;
    }


    int len = ne[n];
    if (len) {
        std::cout << std::string(p + 1, len) << "\n";
    } else {
        std::cout<<"empty\n";
    }
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
