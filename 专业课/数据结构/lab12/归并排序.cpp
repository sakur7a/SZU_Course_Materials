#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

using namespace std;

const int N = 100010;
string a[N], temp[N]; // 将 int 改为 string
int n;

// 核心归并逻辑，保留了你模板中的大部分写法
void merge(int l, int mid, int r) {
    int k = 0, i = l, j = mid + 1;
    while (i <= mid && j <= r) {
        // 题目要求降序(字典序大在前)，所以用 >=
        // 为了保证稳定性，建议当相等时取左边
        if (a[i] >= a[j]) temp[k++] = a[i++];
        else temp[k++] = a[j++];
    }
    while (i <= mid) temp[k++] = a[i++];
    while (j <= r) temp[k++] = a[j++];
    
    // 将排好序的段放回原数组
    for (i = l, j = 0; i <= r; i++, j++)
        a[i] = temp[j];
}

void solve() {
    cin >> n;
    for (int i = 0; i < n; i++) {
        cin >> a[i];
    }

    // --- 改为迭代式归并 ---
    // h 代表当前合并子序列的长度: 1, 2, 4, 8...
    int h = 1;
    while (h < n) {
        int i = 0;
        while (i + 2 * h <= n) {
            merge(i, i + h - 1, i + 2 * h - 1);
            i += 2 * h;
        }
        if (i + h < n) {
            merge(i, i + h - 1, n - 1);
        }

        for (int k = 0; k < n; k++) {
            cout << a[k] << (k == n - 1 ? "" : " ");
        }
        cout << endl;

        // 步长翻倍：1 -> 2 -> 4 -> ...
        h *= 2;
    }
}

int main() {
    // 优化cin/cout速度
    ios::sync_with_stdio(false);
    cin.tie(0);

    int t;
    if (cin >> t) {
        while (t--) {
            solve();
            // 题目要求：每组测试数据的输出之间有1空行
            if (t > 0) cout << endl;
        }
    }
    return 0;
}