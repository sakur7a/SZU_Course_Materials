#include<bits/stdc++.h>
typedef long long LL;
typedef std::pair<int, int> PII;
const int N = 1010, M = 500;

int t, w;
char s[N], p[N];
int ne[55];

void kmp(char s[], char p[]) {
    int res = 0;
    int m = strlen(s + 1), n = strlen(p + 1);
    for (int i = 2, j = 0; i <= n; i++) {
        while (j && p[j + 1] != p[i]) j = ne[j];
        if (p[j + 1] == p[i]) j++;
        ne[i] = j;
    }

    for (int i = 1, j = 0; i <= m; i++) {
        while (j && s[i] != p[j + 1]) j = ne[j];
        if (p[j + 1] == s[i]) j++;
        if (j == n) {
            res++;
            j = ne[j];
        }
    }

    std::cout << p + 1 << ":" << res << std::endl;
}

void solve() {
    std::cin >> t;
    while (t--) {
        std::cin >> s + 1 >> w;
        while (w--) {
            std::cin >> p + 1;
            kmp(s, p);
        }
    }

}


int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    solve();
    return 0;
}
