#include <bits/stdc++.h>
using namespace std;
const int N = 100010;

int t, n, m;
string v[N];
int e[N], ne[N], idx, h[N];


void add(int a, int b) {
    e[idx] = b;
    ne[idx] = h[a];
    h[a] = idx++;
}

void solve() {
    memset(h, -1, sizeof(h));
    idx = 0;
    std::cin >> n >> m;
    while (m--) {
        int a, b;
        std::cin >> a >> b;
        add(a, b), add(b, a);
    }
    int start;
    std::cin >> start;
    std::cout << "BFS from " << start << ":";

    


}
 
int main() {
    solve();
    return 0;
}