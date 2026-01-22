#include <bits/stdc++.h>
using namespace std;

const int N = 1010, INF = 0x3f3f3f3f;

int t, n;
int g[N][N];
bool st[N];
int dist[N];
int pre[N]; 
string names[N]; 
unordered_map<string, int> mp; 

void dijkstra(int start) {
    // 初始化
    memset(dist, 0x3f, sizeof(dist));
    memset(st, 0, sizeof(st));
    memset(pre, -1, sizeof(pre)); 

    dist[start] = 0;

    for (int i = 0; i < n; i++) {
        int t = -1;
        for (int j = 0; j < n; j++) {
            if (!st[j] && (t == -1 || dist[j] < dist[t])) {
                t = j;
            }
        }

        if (t == -1 || dist[t] == INF) break;

        st[t] = true;

        for (int j = 0; j < n; j++) {
            if (g[t][j] != INF) {
                if (dist[t] + g[t][j] < dist[j]) {
                    dist[j] = dist[t] + g[t][j];
                    pre[j] = t;
                }
            }
        }
    }
}


void print_path(int start, int end) {
    vector<int> path;
    int curr = end;
    while (curr != -1) {
        path.push_back(curr);
        curr = pre[curr];
    }
    // 需要倒序输出
    for (int i = path.size() - 1; i >= 0; i--) {
        cout << names[path[i]] << " ";
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (cin >> t) {
        while (t--) {
            cin >> n;
            mp.clear();
            
            for (int i = 0; i < n; i++) {
                cin >> names[i];
                mp[names[i]] = i;
            }

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    int x;
                    cin >> x;

                    if (x == 0) g[i][j] = INF;
                    else g[i][j] = x;
                }
            }

            string start_node_str;
            cin >> start_node_str;
            int start_idx = mp[start_node_str];

            dijkstra(start_idx);

            for (int i = 0; i < n; i++) {

                if (i == start_idx) continue;

                cout << names[start_idx] << "-" << names[i] << "-";

                if (dist[i] == INF) {
                    cout << "-1\n";
                } else {
                    cout << dist[i] << "----[";
                    print_path(start_idx, i);
                    cout << "]\n";
                }
            }
        }
    }
  
    return 0;
}