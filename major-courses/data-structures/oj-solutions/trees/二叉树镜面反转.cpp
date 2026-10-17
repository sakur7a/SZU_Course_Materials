#include <bits/stdc++.h>
using namespace std;
typedef long long LL;
typedef std::pair<int, int> PII;
typedef std::pair<LL, LL> PLL;
const int N = 100010;

int t, n, m;

struct TreeNode {
    char val;
    TreeNode* right;
    TreeNode* left;

    TreeNode(char x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(char x, TreeNode* l, TreeNode* r) : val(x), left(l), right(r) {}
};

TreeNode* build(string s, int& idx) {
    int n = s.size();
    if (idx >= n || s[idx] == '#') {
        idx++;
        return nullptr;
    }

    TreeNode* node = new TreeNode(s[idx++]);
    node->left = build(s, idx);
    node->right = build(s, idx);

    return node;
} 

void preorder(TreeNode* root) {
    if (!root) {
        return;
    }
    std::cout << root->val << " ";
    preorder(root->left);
    preorder(root->right);
}

void inorder(TreeNode* root) {
    if (!root) {
        return;
    }
    inorder(root->left);
    std::cout << root->val << " ";
    inorder(root->right);
}

void postorder(TreeNode* root) {
    if (!root) {
        return;
    }
    postorder(root->left);
    postorder(root->right);
    std::cout << root->val << " ";
}

void levelorder(TreeNode* root) {
    if (root == nullptr) {
        return;
    }
    
    // 模拟队列
    TreeNode* q[N];
    int tail = -1, head = 0;
    q[++tail] = root;

    while (tail >= head) {
        TreeNode* t = q[head];
        head++;
        std::cout << t->val << " ";
        if (t->left) {
            q[++tail] = t->left;
        }
        if (t->right) {
            q[++tail] = t->right;
        }
    }
}

void mirrorReverse(TreeNode* root) {
    if (!root) {
        return;
    }
    TreeNode* temp = root->left;
    root->left = root->right;
    root->right = temp;
    mirrorReverse(root->left);
    mirrorReverse(root->right);
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::cin >> t;
    for (int i = 0; i < t; i++) {
        string s;
        std::cin >> s;
        int idx = 0;
        TreeNode* root = build(s, idx);
        if (!root) {
            std::cout << "NULL\nNULL\nNULL\nNULL\n";
            continue;
        }
        mirrorReverse(root);
        preorder(root);
        std::cout << "\n";
        inorder(root);
        std::cout << "\n";
        postorder(root);
        std::cout << "\n";
        levelorder(root);
        if (i != t - 1) std::cout << "\n";      
    }
  
    return 0;
}
