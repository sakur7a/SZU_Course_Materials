#include <bits/stdc++.h>
using namespace std;
typedef std::pair<int,int> PII;
const int N = 100010;

int n, t;
vector<char> leaves, parents;

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

void collect(TreeNode* node, TreeNode* parent) {
    if(!node) {
        return;
    }

    if (!node->left && !node->right) {
        parents.push_back(parent->val);
        leaves.push_back(node->val);
    }

    collect(node->left, node);
    collect(node->right, node);
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
        leaves.clear();
        parents.clear();
        collect(root, nullptr);
        for (char &l : leaves) {
            std::cout << l << " ";
        }        
        std::cout << "\n";
        for (char &p : parents) {
            std::cout << p << " ";
        }
        std::cout << "\n"; 
    }

    return 0;
}