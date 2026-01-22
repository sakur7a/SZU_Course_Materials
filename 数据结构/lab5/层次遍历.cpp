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


int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::cin >> t;
    while (t--) {
        string s;
        std::cin >> s;
        int idx = 0;
        TreeNode* root = build(s, idx);
        queue<TreeNode*> q;

        q.push(root);
        while (q.size()) {
            int len = q.size();
            for (int i = 0; i < len; i++) {
                auto t = q.front();
                q.pop();

                std::cout<< t->val;

                if (t->left) {
                    q.push(t->left);
                }
                if (t->right) {
                    q.push(t->right);
                }
            }
        }
        std::cout << "\n";
    }

    return 0;
}