#include <iostream>

using namespace std;
const int MAXN = 505;

int adj[MAXN][MAXN];
bool visited[MAXN];

void solve() {
    int n;

    if (!(cin >> n)) return;

    for (int i = 0; i < n; ++i) {
        visited[i] = false;
        for (int j = 0; j < n; ++j) {
            cin >> adj[i][j];
        }
    }

    for (int step = 0; step < n; ++step) {
        int v = -1;

        for (int j = 0; j < n; ++j) {
            if (!visited[j]) {
                bool isZeroInDegree = true;
                for (int i = 0; i < n; ++i) {
                    if (adj[i][j] != 0) {
                        isZeroInDegree = false;
                        break;
                    }
                }
                
                if (isZeroInDegree) {
                    v = j;
                    break;
                }
            }
        }

        if (v != -1) {
            cout << v << " ";
            visited[v] = true;

            for (int k = 0; k < n; ++k) {
                adj[v][k] = 0;
            }
        } else {
            break;
        }
    }
    cout << endl;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(0);

    int t;
    if (cin >> t) {
        while (t--) {
            solve();
        }
    }
    return 0;
}