#include <bits/stdc++.h>
using namespace std;
typedef long long LL;
typedef std::pair<int, int> PII;
typedef std::pair<LL, LL> PLL;
const int N = 100010;

int t, n, m;

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;

    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

void insert(TreeNode* &root, int x) {
    if (!root) {
        root = new TreeNode(x);
        return;
    }

    if (root->val > x) {
        insert(root->left, x);
    } else {
        insert(root->right, x);
    }
}

void inorder(TreeNode* root) {
    if (!root) {
        return;
    }
    
    inorder(root->left);
    std::cout << root->val << " ";
    inorder(root->right);
}

int search(TreeNode* root, int target) {
    int res = 0;
    TreeNode* curr = root;
    while (curr) {
        res++;
        if (curr->val == target) {
            return res;
        } else if (curr->val > target) {
            curr = curr->left;
        } else {
            curr = curr->right;
        }
    }
    return -1;
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::cin >> n;
    TreeNode* root = NULL;
    while (n--) {
        int x;
        std::cin >> x;
        insert(root, x);
    }

    inorder(root);
    std::cout << endl;

    std::cin >> m;
    for (int i = 0; i < m; i++) {
        int val;
        std::cin >> val;
        std::cout << search(root, val) << endl;
    }
  
    return 0;
}
