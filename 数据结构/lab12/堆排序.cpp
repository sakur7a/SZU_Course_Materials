#include <iostream>
#include <vector>
#include <algorithm> // 必须包含，用于 make_heap, pop_heap
#include <functional> // 必须包含，用于 greater

using namespace std;

// 打印数组辅助函数
void printArr(const vector<int>& arr) {
    cout << arr.size();
    for (int x : arr) {
        cout << " " << x;
    }
    cout << endl;
}

void solve() {
    int n;
    while (cin >> n) {
        if (n == 0) continue;
        
        vector<int> arr(n);
        for (int i = 0; i < n; i++) {
            cin >> arr[i];
        }

        // 1. 构建小顶堆 (使用 greater)
        // make_heap 默认是大顶堆，加了 greater 后变成小顶堆
        make_heap(arr.begin(), arr.end(), greater<int>());
        
        // 输出初始建堆结果
        printArr(arr);

        // 2. 模拟堆排序过程
        // pop_heap 的作用是：将堆顶元素(arr.front())移到区间末尾(arr.back())，
        // 然后对剩余部分重新进行下沉调整使其保持堆性质。
        // 这完全符合题目要求的“交换”+“筛选”。
        for (int i = 0; i < n - 1; i++) {
            // 对当前范围 [begin, end - i) 进行 pop 操作
            pop_heap(arr.begin(), arr.end() - i, greater<int>());
            
            // 输出本趟结果
            printArr(arr);
        }
    }
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    solve();
    return 0;
}