#include <iostream>
#include <vector>
#include <string>
#include <map>

using namespace std;

// 定义堆的最大容量
const int MAXN = 1005;
int h[MAXN]; // 堆数组
int n;       // 堆中元素个数

// 向上调整（插入时使用）
void up(int u) {
    while (u / 2 > 0 && h[u] < h[u / 2]) {
        swap(h[u], h[u / 2]);
        u /= 2;
    }
}

// 插入操作
void insert(int x) {
    n++;
    h[n] = x;
    up(n);
}

// 查找元素的下标
int find_index(int x) {
    for (int i = 1; i <= n; i++) {
        if (h[i] == x) return i;
    }
    return -1; // 理论上不会执行，题目保证存在
}

int main() {
    // 优化 I/O
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int N, M;
    if (cin >> N >> M) {
        n = 0; // 初始化堆大小
        for (int i = 0; i < N; i++) {
            int val;
            cin >> val;
            insert(val);
        }

        // 处理 M 个查询
        for (int i = 0; i < M; i++) {
            int x, y;
            string s;
            cin >> x;      // 读取第一个数字 x
            cin >> s;      // 读取 x 后面的第一个单词

            bool flag = false;

            if (s == "and") {
                // 格式: x and y are siblings
                cin >> y;
                string temp;
                cin >> temp >> temp; // 吃掉 "are" "siblings"
                
                int idx_x = find_index(x);
                int idx_y = find_index(y);
                // 兄弟节点：父节点索引相同，且不是同一个节点
                if (idx_x / 2 == idx_y / 2 && idx_x != idx_y) flag = true;

            } else {
                // 此时 s 是 "is"
                cin >> s; // 读取 "is" 后面的词
                if (s == "a") {
                    // 格式: x is a child of y
                    string temp;
                    cin >> temp >> temp >> y; // 吃掉 "child" "of", 读取 y
                    
                    int idx_x = find_index(x);
                    int idx_y = find_index(y);
                    // x 是 y 的孩子：x 的父节点是 y
                    if (idx_x / 2 == idx_y) flag = true;

                } else {
                    // 此时 s 是 "the"
                    cin >> s; // 读取 "the" 后面的词
                    if (s == "root") {
                        // 格式: x is the root
                        int idx_x = find_index(x);
                        if (idx_x == 1) flag = true;

                    } else if (s == "parent") {
                        // 格式: x is the parent of y
                        string temp;
                        cin >> temp >> y; // 吃掉 "of", 读取 y
                        
                        int idx_x = find_index(x);
                        int idx_y = find_index(y);
                        // x 是 y 的父亲：y 的父节点是 x
                        if (idx_y / 2 == idx_x) flag = true;
                    }
                }
            }

            cout << (flag ? "T" : "F") << endl;
        }
    }
    return 0;
}