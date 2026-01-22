#include <iostream>
#include <string>
#include <algorithm> // 用于 max 函数
#include <vector>

using namespace std;

// 递归函数：执行后序遍历，计算高度并输出平衡因子
// 参数：u - 当前节点的数组下标, n - 数组总大小, tree - 存储树的字符串
// 返回值：当前以 u 为根的子树的高度
int getBalanceAndHeight(int u, int n, const string& tree) {
    // 1. 基准情况：下标越界 或 节点值为 '0'，视为空树
    if (u >= n || tree[u] == '0') {
        return 0; // 空树高度为 0
    }

    // 2. 递归遍历左子树 (2*u + 1)
    int leftHeight = getBalanceAndHeight(2 * u + 1, n, tree);
    
    // 3. 递归遍历右子树 (2*u + 2)
    int rightHeight = getBalanceAndHeight(2 * u + 2, n, tree);

    // 4. 处理根节点（当前节点）
    // 计算平衡因子
    int balanceFactor = leftHeight - rightHeight;
    
    // 输出：节点字符 和 平衡因子
    // 因为是在左右子树递归完成后输出，所以符合后序遍历顺序
    cout << tree[u] << " " << balanceFactor << endl;

    // 5. 返回当前节点高度
    return max(leftHeight, rightHeight) + 1;
}

void solve() {
    int n;
    // 读取节点个数
    if (!(cin >> n)) return;
    
    // 读取树的字符序列
    // 使用循环读取字符比较稳健，既能处理连续字符串 "ABC00D"，也能处理带空格的 "A B C 0 0 D"
    string tree;
    tree.resize(n);
    for (int i = 0; i < n; ++i) {
        cin >> tree[i];
    }

    // 从根节点（下标0）开始递归
    getBalanceAndHeight(0, n, tree);
}

int main() {
    // 优化 IO 效率
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int t;
    // 读取测试组数
    if (cin >> t) {
        while (t--) {
            solve();
        }
    }
    return 0;
}