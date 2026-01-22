#include <bits/stdc++.h>
using namespace std;
typedef long long LL;
typedef std::pair<int, int> PII;
typedef std::pair<LL, LL> PLL;
const int N = 100010;

int t, n, m;

struct TreeNode {
    int weight;
    TreeNode* left;
    TreeNode* right;

    TreeNode(int w) : weight(w), left(nullptr), right(nullptr) {}
};

TreeNode* build(string s, vector<int> w, int &idx, int &wIdx) {
    if (idx >= s.size() || s[idx] == '0') {
        idx++;
        return nullptr;
    }

    TreeNode* node = new TreeNode(w[wIdx++]);
    idx++;
    node->left = build(s, w, idx, wIdx);
    node->right = build(s, w, idx, wIdx);

    return node;
}

void dfs(TreeNode* node, int current, int &mx) {
    if (!node) {
        return;
    }
    current += node->weight;
    if (!node->left && !node->right) {
        mx = max(mx, current);
        return;
    }
    dfs(node->left, current, mx);
    dfs(node->right, current, mx);
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::cin >> t;
    while (t--) {
        string pre;

        std::cin >> pre >> n;
        vector<int> w(n);
        for (int i = 0; i < n; i++) {
            std::cin >> w[i];
        }
        int idx = 0, wIdx = 0;
        TreeNode* root = build(pre, w, idx, wIdx);
        int ans = INT_MIN;
        dfs(root, 0, ans);
        std::cout << ans << "\n";
    }
  
    return 0;
}
