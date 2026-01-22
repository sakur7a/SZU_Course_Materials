#include<bits/stdc++.h>
typedef long long LL;
typedef std::pair<int,int> PII;
const int N = 110;

int g[N][N];
bool st[N][N];
int t, n;
int dx[] = {0, 1, -1, 0}, dy[] = {1, 0, 0, -1};
std::stack<PII> path;


bool dfs(int x, int y) {
    if (x == n - 1 && y == n - 1) {
        path.push({x, y});
        return true;
    }

    st[x][y] = true;
    path.push({x, y});
    for (int i = 0 ; i < 4; i++) {
        int X = x + dx[i], Y = y + dy[i];
        if (X >= 0 && X < n && Y >= 0 && Y < n && !st[X][Y] && g[X][Y] == 0) {
            if (dfs(X, Y)) {
                return true;
            }
        }
        
    }

    path.pop();
    return false;
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
  
    std::cin>>t;
    while (t--) {
        std::cin>>n;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                std::cin>>g[i][j];
                st[i][j] = false;
            }
        }

        while (path.size()) {
            path.pop();
        }

        if (g[0][0] == 0 && dfs(0, 0)) {
            std::stack<PII> temp;
            while (path.size()) {
                temp.push(path.top());
                path.pop();
            }
            int i = 0;
            while (temp.size()) {
                auto top = temp.top();
                temp.pop();
                if ((++i) % 4 == 0) {
                    std::cout << '[' << top.first << ',' << top.second << ']' << "--" << std::endl;
                } else {
                    std::cout << '[' << top.first << ',' << top.second << ']' << "--";
                } 
            }
            std::cout << "END" << std::endl;
        } else {
            std::cout<<"no path\n";
        }
    }
    return 0;
}
