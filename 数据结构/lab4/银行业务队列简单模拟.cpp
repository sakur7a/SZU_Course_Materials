#include<bits/stdc++.h>
using namespace std;
typedef long long LL;
typedef std::pair<int, int> PII;
const int N = 100010;

int n;
queue<int> a, b;
vector<int> ans;

void solve() {
    std::cin >> n;
    for (int i = 0 ; i < n; i++) {
        int x;
        std::cin >> x;
        if (x % 2 == 1) {
            a.push(x);
        } else {
            b.push(x);
        }
    }

    while (a.size() || b.size()) {
        for (int i = 0; i < 2; i++) {
            if (a.size()) {
                ans.push_back(a.front());
                a.pop();
            }
        }
        if (b.size()) {
            ans.push_back(b.front());
            b.pop();
        }
    }

    for (int i = 0; i < ans.size(); i++) {
        if (i) std::cout << " ";
        std::cout << ans[i];
    }
}



int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
  
    solve();
    return 0;
}
