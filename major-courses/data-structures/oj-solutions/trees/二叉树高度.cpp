#include <bits/stdc++.h>
using namespace std;
typedef long long LL;
typedef std::pair<int, int> PII;
typedef std::pair<int, int> PII;
const int N = 100010;

int n, t;


struct TreeNode {
    char val;
    TreeNode* left;
    TreeNode* right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(char x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(char x, TreeNode* l, TreeNode* r) : val(x), left(l), right(r) {}
};


TreeNode* build(const string& pre, int& idx) {
    if (idx == pre.size() || pre[idx] == '0') {
        idx++;
        return nullptr;
    }

    TreeNode* node = new TreeNode(pre[idx++]);
    node->left = build(pre, idx);
    node->right = build(pre, idx);

    return node;
}

int height(TreeNode* root) {
    if (!root) {
        return 0;
    }

    return 1 + max(height(root->left), height(root->right));
}


int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::cin >> t;
    while (t--) {
        string s;
        std::cin >> s;
        int idx = 0;
        TreeNode* root = build(s, idx);

        std::cout << height(root) << "\n";
    }

    return 0;
}