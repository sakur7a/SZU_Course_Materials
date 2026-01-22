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


    cin >> n;
    vector<char> values(n);
    for (int i = 0; i < n; ++i) {
        cin >> values[i];
    }
    
    vector<int> parents(n);
    for (int i = 0; i < n; ++i) {
        cin >> parents[i];
    }
    
    vector<vector<int>> children(n);
    for (int i = 0; i < n; ++i) {
        int parent = parents[i];
        if (parent != -1) { 
            children[parent].push_back(i);
        }
    }

    for (int i = 0; i < n; ++i) {
        cout << values[i] << " ";  
        if (children[i].empty()) {  
            cout << "-1 ";
        } else {  
            for (int child : children[i]) {
                cout << child << " ";
            }
        }
        cout << endl;
    }
  
    return 0;
}
