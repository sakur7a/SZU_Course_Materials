#include <bits/stdc++.h>
using namespace std;
typedef long long LL;
typedef std::pair<int, int> PII;
typedef std::pair<LL, LL> PLL;
const int N = 100010;

int t, n, m;
string pre, in;
unordered_map<int, int> mp;

struct TreeNode {
    TreeNode* left;
    TreeNode* right;
    char val;

    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(char x, TreeNode* l, TreeNode* r) : val(x), left(l), right(r) {}
};

TreeNode* dfs(int pre_l, int pre_r, int in_l, int in_r) {
    if (pre_r == pre_l) {
            return nullptr;
        }

        int root_val = pre[pre_l];
        int len = mp[root_val] - in_l;

        TreeNode* left = dfs(pre_l + 1, pre_l + 1 + len, in_l, in_l + len);
        TreeNode* right = dfs(pre_l + 1 + len, pre_r, in_l + len + 1, in_r);

        return new TreeNode(root_val, left, right);
    };

int getHeight(TreeNode* root) {
    if (!root) {
        return 0;
    }
    return 1 + max(getHeight(root->left), getHeight(root->right));
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::cin >> n >> pre >> in;

    for (int i = 0; i < n; i++) {
        mp[in[i]] = i;
    }

    TreeNode* root = dfs(0, n, 0, n);
    std::cout << getHeight(root);

  
    return 0;
}
