#include <bits/stdc++.h>
using namespace std;
typedef long long LL;
typedef std::pair<int, int> PII;
typedef std::pair<LL, LL> PLL;
const int N = 100010;

int t, n, m, k;
vector<int> a(N);

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::cin >> t;
    while (t--) {
        std::cin >> m >> n;
        vector<int> hashmap(m, -1);
        for (int i = 0; i < n; i++) {
            int key;
            std::cin >> key;
            int pos = key % 11;
            while (hashmap[pos] != -1) {
                pos = (pos + 1) % m;
            }
            hashmap[pos] = key;
        }

        for (int i = 0; i < m; i++) {
            if (i) {
                std::cout << " ";
            }
            if (hashmap[i] != -1) {
                std::cout << hashmap[i];
            } else {
                std::cout << "NULL";
            }
        }
        std::cout << "\n";
        

        std::cin >> k;
        while (k--) {
            int target;
            std::cin >> target;
            int pos = target % 11;
            int cnt = 0, findpos = -1;
            bool flag = false;
            for (int i = 0; i < m; i++) {
                cnt++;
                if (hashmap[pos] == -1) {
                    break;
                } else if (hashmap[pos] == target) {
                    flag = true;
                    findpos = pos + 1;
                    break;
                }
                pos = (pos + 1) % m;
            }

            if (flag) {
                cout << "1 " << cnt << " " << findpos << "\n";
            } else {
                cout << "0 " << cnt << "\n";
            }
        }
    }
  
    return 0;
}
