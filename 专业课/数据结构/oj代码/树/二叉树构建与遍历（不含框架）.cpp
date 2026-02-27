#include <bits/stdc++.h>
using namespace std;
typedef std::pair<int,int> PII;
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
    if (idx == pre.size() || pre[idx] == '#') {
        idx++;
        return nullptr;
    }

    TreeNode* node = new TreeNode(pre[idx++]);
    node->left = build(pre, idx);
    node->right = build(pre, idx);

    return node;
}

void preorder(TreeNode* root) {
    if (!root) {
        return;
    }
    std::cout << root->val;
    preorder(root->left);
    preorder(root->right);
}

void inorder(TreeNode* root) {
    if (!root) {
        return;
    }
    
    inorder(root->left);
    std::cout << root->val;
    inorder(root->right);
}

void postorder(TreeNode* root) {
    if (!root) {
        return;
    }
    
    postorder(root->left);
    postorder(root->right);
    std::cout << root->val;
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
        preorder(root);
        std::cout << "\n";
        inorder(root);
        std::cout << "\n";
        postorder(root);
        std::cout << "\n";
    }
    
    return 0;
}