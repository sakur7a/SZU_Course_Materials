#include <bits/stdc++.h>
using namespace std;
typedef pair<int, int> PII;
typedef long long LL;
const int N = 100010;

int n;


struct TreeNode {
    int parent, left, right, weight;
} nodes[N];

int main() {
    std::cin >> n;
    for (int i = 1; i <= n; i++) {
        std::cin >> nodes[i].weight;
    }

    priority_queue<PII, vector<PII>, greater<PII>> heap;
    for (int i = 1; i <= n; i++) {
        heap.push({nodes[i].weight, i});
    }
    int k = n + 1;

    while (heap.size() > 1) {
        auto [w1, m1] = heap.top();
        heap.pop();
        auto [w2, m2] = heap.top();
        heap.pop();

        nodes[k].weight = w1 + w2;
        nodes[k].left = m1, nodes[k].right = m2;
        nodes[m1].parent = k, nodes[m2].parent = k;
        heap.push({nodes[k].weight, k});
        k++;
    }

    for (int i = 1; i <= n; i++) {
        string code;
        int cur = i;
        while (nodes[cur].parent != 0) {
            int p = nodes[cur].parent;
            if (nodes[p].left == cur) {
                code += "0";
            } else {
                code += "1";
            }
            cur = p;
        }
        reverse(code.begin(), code.end());
        std::cout << nodes[i].weight << "-" << code << "\n";
    }

    return 0;
}