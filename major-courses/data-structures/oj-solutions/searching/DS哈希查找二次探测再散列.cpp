#include <bits/stdc++.h>
using namespace std;
typedef long long LL;
typedef std::pair<int, int> PII;
typedef std::pair<LL, LL> PLL;
const int N = 100010;

int t, n, m, k;

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
            int d0 = key % 11;

            if (hashmap[d0] == -1) {
                hashmap[d0] = key;
            } else {
                for (int w = 1; w <= m; w++) {
                    int d1 = (d0 + w * w) % m;
                    if (hashmap[d1] == -1) {
                        hashmap[d1] = key;
                        break;
                    }

                    int d2 = (d0 - w * w) % m;
                    if (d2 < 0) d2 += m;
                    if (hashmap[d2] == -1) {
                        hashmap[d2] = key;
                        break;
                    }
                }
            }
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
            int d0 = target % 11;
            int cnt = 0, findpos = -1;
            bool flag = false;

            cnt++;
            if (hashmap[d0] == target) {
                flag = true;
                findpos = d0 + 1;
            } else {
                for (int w = 1; w <= m; w++) {
                    int d1 = (d0 + w * w) % m;
                    cnt++;
                    if (hashmap[d1] == target) {
                        flag = true;
                        findpos = d1 + 1;
                        break;
                    }
                    if (hashmap[d1] == -1) {
                        break; 
                    }

                    int d2 = (d0 - w * w) % m;
                    if (d2 < 0) d2 += m;
                    cnt++;
                    if (hashmap[d2] == target) {
                        flag = true;
                        findpos = d2 + 1;
                        break;
                    }
                    if (hashmap[d2] == -1) {
                        break; 
                    }
                }
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
