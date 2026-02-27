#include <bits/stdc++.h>
using namespace std;
typedef long long LL;
typedef std::pair<int, int> PII;
typedef std::pair<LL, LL> PLL;
const int N = 100010;

int t, n, m;

void preorder(int node, vector<char> &values, vector<vector<int>> &childrens) {
    std::cout << values[node];
    for (auto &child : childrens[node]) {
        preorder(child, values, childrens);
    }
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::cin >> t;
    while (t--) {
        std::cin >> n;
        vector<char> values(n);

        for (int i = 0; i < n; i++) {
            std::cin >> values[i];
        }

        vector<int> parents(n);
        int root = -1;
        for (int i = 0; i < n; i++) {
            std::cin >> parents[i];
            if (parents[i] == -1) {
                root = i;
            }
        }

        vector<vector<int>> childrens(n);
        for (int i = 0; i < n; i++) {
            int p = parents[i];
            if (p != -1) {
                childrens[p].push_back(i);
            }
        }

        preorder(root, values, childrens);
        
        cout << endl;
    }
  
    return 0;
}
