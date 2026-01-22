#include <bits/stdc++.h>
using namespace std;

#define ok 1
#define error -1

struct TreeNode {
    char ch;
    int weight;
    int parent;
    int left;
    int right;
};

const int MAXN = 100010;
TreeNode nodes[MAXN * 2];
int rootIdx;
int totalNodes;

int decode(const string& codestr, char txtstr[]) {
    if (!rootIdx) return error;
    int p = rootIdx;
    string result;
    for (size_t i = 0; i < codestr.size(); ++i) {
        char ch = codestr[i];
        if (ch == '0') {
            if (nodes[p].left == 0) return error;
            p = nodes[p].left;
        } else if (ch == '1') {
            if (nodes[p].right == 0) return error;
            p = nodes[p].right;
        } else {
            return error;
        }
        if (nodes[p].left == 0 && nodes[p].right == 0) {
            result.push_back(nodes[p].ch);
            p = rootIdx;
        } else if (i + 1 == codestr.size() && nodes[p].left != 0 && nodes[p].right != 0) {
            return error;
        }
    }
    if (p != rootIdx) return error;
    for (size_t i = 0; i < result.size(); ++i) txtstr[i] = result[i];
    txtstr[result.size()] = '\0';
    return ok;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    cin >> n;
    for (int i = 1; i <= 2 * n; ++i) nodes[i] = TreeNode();

    for (int i = 1; i <= n; ++i) {
        cin >> nodes[i].weight;
    }
    for (int i = 1; i <= n; ++i) {
        char ch;
        cin >> ch;
        nodes[i].ch = ch;
    }

    priority_queue<pair<int,int>, vector<pair<int,int>>, greater<pair<int,int>>> minHeap;
    for (int i = 1; i <= n; ++i) minHeap.emplace(nodes[i].weight, i);

    int k = n + 1;
    while (minHeap.size() > 1) {
        auto [w1, idx1] = minHeap.top(); minHeap.pop();
        auto [w2, idx2] = minHeap.top(); minHeap.pop();

        nodes[k] = TreeNode();
        nodes[k].weight = w1 + w2;
        nodes[k].left = idx1, nodes[k].right = idx2;
        nodes[idx1].parent = k, nodes[idx2].parent = k;

        minHeap.emplace(nodes[k].weight, k);
        ++k;
    }
    rootIdx = minHeap.top().second;
    totalNodes = k - 1;

    int q;
    cin >> q;
    string code;
    while (q--) {
        cin >> code;
        char decoded[20005];
        if (decode(code, decoded) == ok) cout << decoded << "\n";
        else cout << "error\n";
    }
    return 0;
}