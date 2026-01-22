#include<bits/stdc++.h>
typedef long long LL;
typedef std::pair<int, int> PII;
const int N = 100010;

int t, n, k;
std::stack<char> stk;

char toChar(int x) {
    if (x < 10) {
        return '0' + x;
    }
    return 'A' + (x - 10);
}

void solve() {
    std::cin >> t;
    while (t--) {
        std::cin >> n >> k;

        if (n == 0) {
            stk.push('0');
        } else {
            while (n) {
                stk.push(toChar(n % k));
                n /= k;
            }
        }

        while (stk.size()) {
            std::cout << stk.top();
            stk.pop();
        }

        std::cout << "\n";
    }
}


int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    solve();
    return 0;
}
