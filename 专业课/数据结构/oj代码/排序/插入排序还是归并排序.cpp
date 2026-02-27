#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int main() {
    // 优化 I/O 速度
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n;
    if (cin >> n) {
        vector<int> a(n); // 原始序列
        vector<int> b(n); // 中间序列

        for (int i = 0; i < n; i++) cin >> a[i];
        for (int i = 0; i < n; i++) cin >> b[i];

        // --- 1. 检查是否为插入排序 ---
        // 寻找中间序列 b 的最长有序前缀
        // i 指向有序部分的最后一个元素的下标
        int i = 0;
        while (i < n - 1 && b[i] <= b[i + 1]) {
            i++;
        }
        // 此时，0 到 i 是有序的。
        // 检查 i+1 到 n-1 部分是否与原始序列 a 相同
        bool isInsertion = true;
        for (int j = i + 1; j < n; j++) {
            if (a[j] != b[j]) {
                isInsertion = false;
                break;
            }
        }

        if (isInsertion) {
            cout << "Insertion Sort" << endl;
            // 插入排序的下一步：将 b[i+1] 这个元素纳入有序序列
            // 即对前 i+2 个元素进行排序
            // 注意边界：如果 i+2 > n，则 sort 到 end() 即可
            int sort_end = i + 2;
            if (sort_end > n) sort_end = n;
            
            sort(b.begin(), b.begin() + sort_end);
            
            // 输出结果
            for (int k = 0; k < n; k++) {
                cout << b[k] << (k == n - 1 ? "" : " ");
            }
            cout << endl;
        } 
        else {
            cout << "Merge Sort" << endl;
            // --- 2. 模拟归并排序 ---
            // 从原始序列 a 开始模拟，直到与 b 相等
            int k = 1; // 初始步长（子序列长度）
            bool match = false;

            while (!match) {
                // 判断当前模拟状态是否与 b 相等
                // 注意：由于我们是先模拟再判断，且题目保证 b 是中间序列，
                // 所以我们先比对，如果不等再进行归并操作？
                // 不，通常 b 既然是“中间序列”，它肯定是由 a 经过至少一次归并得到的。
                // 所以我们先做一次归并，再比对。
                
                // 执行一趟归并：步长为 k，每 2*k 个元素一组进行排序
                // 使用 a 进行模拟
                for (int j = 0; j < n; j += 2 * k) {
                    // 计算当前段的结束位置，防止越界
                    int end = min(j + 2 * k, n);
                    sort(a.begin() + j, a.begin() + end);
                }
                
                // 此时 a 已经是步长为 k 的归并结果（block size = 2*k）
                // 题目定义的归并迭代是：归并两个相邻的有序子序列。
                // 初始看作 N 个长为 1 的序列。
                // 第一次迭代：k=1，归并成长为 2 的序列。
                // 所以上面的 sort 操作正好对应了一次迭代。
                
                // 比较 a 和 b
                if (a == b) {
                    match = true;
                    // 如果匹配，再进行一轮迭代
                    k *= 2;
                    for (int j = 0; j < n; j += 2 * k) {
                        int end = min(j + 2 * k, n);
                        sort(a.begin() + j, a.begin() + end);
                    }
                    // 输出结果
                    for (int z = 0; z < n; z++) {
                        cout << a[z] << (z == n - 1 ? "" : " ");
                    }
                    cout << endl;
                }
                
                // 步长翻倍，准备下一次模拟（如果还没匹配）
                k *= 2;
            }
        }
    }

    return 0;
}