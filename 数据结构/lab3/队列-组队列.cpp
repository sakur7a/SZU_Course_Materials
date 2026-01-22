#include <bits/stdc++.h>
typedef long long LL;
typedef std::pair<int,int> PII;
const int N = 100010;

int t, n, x;
std::unordered_map<int, int> mp;
std::queue<int> group_queue;
std::vector<std::queue<int>> queues;


void solve() {
    std::cin >> t;

    std::vector<bool> st(t, false);

    for (int i = 0; i < t; i++) {
        std::cin >> n;
        while (n--) {
            std::cin>>x;
            mp[x] = i;
        }
    }

    std::string op;
    while (std::cin>>op && op != "STOP") {
        if (op == "ENQUEUE") {
            int a;
            std::cin>>a;
            int g = mp[a];
            queues[g].push(a);
            if (!st[g]) {
                group_queue.push(g);
                st[g] - true;
            }
        } else if (op == "DEQUEUE") {
            int g = group_queue.front();
            int w = queues[g].front();
            queues[g].pop();
            std::cout << w << " ";
            if (queues[g].empty()) {
                group_queue.pop();
                st[g] = false;
            }
        }
    }
}



int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    solve();
    return 0;
}