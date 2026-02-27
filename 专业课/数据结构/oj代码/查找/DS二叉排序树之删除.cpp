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

TreeNode* findMin(TreeNode* root) {
    while (root->left) {
        root = root->left;
    }
    return root;
}

void deleteNode(TreeNode* &root, int key) {
    if (!root) {
        return;
    }

    if (key < root->val) {
        deleteNode(root->left, key);
    } else if (key > root->val) {
        deleteNode(root->right, key); 
    } else {
        if (!root->left) {
            TreeNode* temp = root;
            root = root->right;
            delete temp;
        } else if (!root->right) {
            TreeNode* temp = root;
            root = root->left;
            delete temp;
        } else {
            TreeNode* temp = findMin(root->right);
            root->val = temp->val;
            deleteNode(root->right, temp->val);
        }
    }
    
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::cin >> t;
    while (t--) {
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
            int key;
            std::cin >> key;
            deleteNode(root, key);
            
            inorder(root);
            std::cout << endl;
        }
    }
    
  
    return 0;
}
