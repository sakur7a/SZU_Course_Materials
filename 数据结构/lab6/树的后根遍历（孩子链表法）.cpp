#include <bits/stdc++.h>
using namespace std;
typedef long long LL;
typedef std::pair<int, int> PII;
typedef std::pair<LL, LL> PLL;
const int N = 100010;

int t, n, m;

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int root;
    std::cin >> n >> root;
    
    vector<char> nodeValues(n);       
    vector<vector<int>> children(n); 
    
    std::function<void(int)> postOrder = [&](int node) {
        for (int child : children[node]) {
            postOrder(child);
        }
        cout << nodeValues[node];
    };
    
    for (int i = 0; i < n; ++i) {
        char val;
        std::cin >> val;
        nodeValues[i] = val;
        
        int child;
        while (cin >> child) {
            if (child == -1) {
                break;
            }
            children[i].push_back(child);
        }
    }
    

    postOrder(root);
    cout << endl;
  
    return 0;
}
