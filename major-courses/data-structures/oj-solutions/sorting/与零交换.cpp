#include <iostream>
#include <vector>

using namespace std;

int main() {
    // 优化标准I/O效率
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n;
    if (cin >> n) {
        vector<int> a(n);
        // 读取排列
        for(int i = 0; i < n; i++) {
            cin >> a[i];
        }

        // visited数组用于标记元素是否已被处理过
        vector<bool> visited(n, false);
        int total_swaps = 0;

        for(int i = 0; i < n; i++) {
            // 如果该元素已被处理，或者是已经排好序的孤立点(arr[i] == i)，则跳过
            if (visited[i] || a[i] == i) {
                visited[i] = true;
                continue;
            }

            // 发现一个新的环，开始遍历
            int count = 0;
            bool hasZero = false;
            int curr = i;
            
            // 沿着 i -> a[i] -> a[a[i]] ... 的路径遍历直到回到原点
            while (!visited[curr]) {
                visited[curr] = true;
                if (curr == 0) { // 检查环中是否包含索引 0
                    hasZero = true;
                }
                curr = a[curr]; // 移动到下一个位置
                count++;
            }
            
            // 根据策略累加交换次数
            if (hasZero) {
                // 含0环：需要 K-1 次交换
                total_swaps += (count - 1);
            } else {
                // 不含0环：需要先借0进来(1次)再归位(count次)，共 count+1 次
                total_swaps += (count + 1);
            }
        }

        cout << total_swaps << endl;
    }
    return 0;
}