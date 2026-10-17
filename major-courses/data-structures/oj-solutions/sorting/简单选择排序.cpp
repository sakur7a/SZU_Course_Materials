#include <iostream>
#include <vector>
#include <algorithm> // 用于 swap 函数

using namespace std;

void solve() {
    int n;
    // 读取数组大小
    if (!(cin >> n)) return;

    // 读取数组元素
    vector<int> a(n);
    for (int i = 0; i < n; i++) {
        cin >> a[i];
    }

    // 简单选择排序核心逻辑
    // 外层循环控制趟数，共需 n-1 趟
    for (int i = 0; i < n - 1; i++) {
        int min_idx = i; // 假设当前位置 i 就是最小值所在位置

        // 内层循环：在无序区 [i+1 ... n-1] 中找实际最小值的下标
        for (int j = i + 1; j < n; j++) {
            if (a[j] < a[min_idx]) {
                min_idx = j;
            }
        }

        // 将找到的最小值与当前位置 i 的元素交换
        // 即使 min_idx == i (即当前位置已经是最小)，通常也视作一次交换逻辑(或不换)，
        // 但题目要求每趟必输出，且逻辑上这代表第 i 个位置确定了。
        if (min_idx != i) {
            swap(a[i], a[min_idx]);
        }

        // 输出本趟排序后的结果
        for (int k = 0; k < n; k++) {
            cout << a[k] << (k == n - 1 ? "" : " ");
        }
        cout << endl;
    }
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int t;
    if (cin >> t) {
        while (t--) {
            solve();
            if (t > 0) {
                cout << endl;
            }
        }
    }
    return 0;
}