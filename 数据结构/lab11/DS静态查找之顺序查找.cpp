#include <bits/stdc++.h>
using namespace std;
typedef long long LL;
typedef std::pair<int, int> PII;
typedef std::pair<LL, LL> PLL;
const int N = 100010;

int t, n, m;

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::cin >> n;
    vector<int> data(n + 1);
    for (int i = 1; i <= n; i++) {
        std::cin >> data[i];
    }

    std::cin >> t;
    while (t--) {
        int target;
        std::cin >> target;
        data[0] = target;
        int i = n;    
        while (data[i] != target) {
            i--;
        }

        if (i == 0) {
            cout << "error\n";
        } else {
            cout << i << "\n";
        }
    }

    return 0;
}
