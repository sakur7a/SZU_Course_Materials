#include <bits/stdc++.h>
using namespace std;
typedef std::pair<int,int> PII;
const int N = 100010;

int t, n;

void preorder(int i, vector<int>& a) {
    if (i > n) {
        return;
    }
    int val = a[i];
    if (val == 0) {
        return;
    }
    std::cout << val << " ";
    preorder(2 * i, a);
    preorder(2 * i + 1, a);
}


int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::cin >> t;
    while (t--) {
        std::cin >> n;
        vector<int> a(n + 1);
        for (int i = 1;i <= n; i++) {
            std::cin >> a[i];
        }
        preorder(1, a);
        std::cout << "\n";
    }


    
    return 0;
}