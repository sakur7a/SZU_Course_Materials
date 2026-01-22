#include <bits/stdc++.h>
using namespace std;
typedef long long LL;
typedef std::pair<int, int> PII;
typedef std::pair<LL, LL> PLL;
const int N = 100010;

int t, n, m;

struct TreeNode {
    char val;
    TreeNode* left;
    TreeNode* right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(char x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(char x, TreeNode* l, TreeNode* r) : val(x), left(l), right(r) {}
};

TreeNode* build(string s, int &idx){
    if (idx == s.size() || s[idx] == '#') {
        idx++;
        return nullptr;
    }
    TreeNode* node = new TreeNode(s[idx++]);
    node->left = build(s, idx);
    node->right = build(s, idx);
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

        vector<char> nodes;
        vector<int> parents;
        queue<pair<TreeNode*, int>> q;
        if (root) q.push({root, -1});

        while (q.size()) {
            auto [node, parentIdx] = q.front();
            q.pop();

            int curIdx = nodes.size();
            nodes.push_back(node->val);
            parents.push_back(parentIdx);


            if (node->left) {
                q.push({node->left, curIdx});
            }
            if (node->right) {
                q.push({node->right, curIdx});
            }
            
        }

        for (int i = 0; i < nodes.size(); i++) {
            if (i > 0) std::cout << " ";
            std::cout << nodes[i];
        }
        std::cout << std::endl;
        for (int i = 0; i < parents.size(); i++) {
            if (i > 0) std::cout << " ";
            std::cout << parents[i];
        }

        std::cout << std::endl;
        
    }
  
    return 0;
}
