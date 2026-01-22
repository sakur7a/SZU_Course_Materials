#include <bits/stdc++.h>
using namespace std;
typedef long long LL;
typedef std::pair<int, int> PII;
typedef std::pair<LL, LL> PLL;
const int N = 100010;

int t, n, m;
int a[N];

void solve() {
    std::cin >> n;
    for (int i = 0; i < n; i++) {
        std::cin >> a[i];
    } 

    for (int gap = n / 2; gap > 0; gap /= 2) {
        for (int i = gap; i < n; i++) {
            int temp = a[i];
            int j;
            for (j = i; j >= gap && a[j - gap] < temp; j -= gap) {
                a[j] = a[j - gap];
            }
            a[j] = temp;
        }


        for (int i = 0; i < n; i++) {
            if (i) {
                std::cout << " ";
            }
            std::cout << a[i];
        }
        std::cout << "\n";
    }

}


int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::cin >> t;
    for (int i = 0; i < t; i++) {
        if (i) {
            std::cout << "\n";
        }
        solve();
    }
  
    return 0;
}
