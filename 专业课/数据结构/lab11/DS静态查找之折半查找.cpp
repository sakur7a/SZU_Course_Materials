#include <bits/stdc++.h>
using namespace std;
typedef long long LL;
typedef std::pair<int, int> PII;
typedef std::pair<LL, LL> PLL;
const int N = 100010;

int t, n, m;
vector<int> a(N);

void binary(int x) {
    int l = 1, r = n;
    while (l < r) {
        int mid = (l + r) >> 1;
        if (a[mid] >= x) {
            r = mid;
        } else {
            l = mid + 1;
        }
    }

    if (a[l] == x) {
        std::cout << l << "\n";
    } else {
        std::cout << "error\n";
    }
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::cin >> n;

    for (int i = 1; i <= n; i++) {
        std::cin >> a[i];
    }

    std::cin >> t;
    while (t--) {
        int target;
        std::cin >> target;
        binary(target);
    }

    return 0;
}
