#include<bits/stdc++.h>
using namespace std;
typedef long long LL;
const int N = 100010;
typedef pair<long long, long long> pll;


long long pow(int a, int m, int mod) {
    long long ans = 1;
    long long contribute = a;
    while (m > 0) {
        if (m % 2 == 1) {
            ans = ans * contribute % mod;
            if (ans < 0) {
                ans += mod;
            }
        }
        contribute = contribute * contribute % mod;
        if (contribute < 0) {
            contribute += mod;
        }
        m /= 2;
    }
    return ans;
}

int check(const vector<int> & arr, int m, int a1, int a2, int mod1, int mod2) {
    int n = arr.size();
    // 若子串长度的2倍超过字符串长度，不可能有不重叠的两个子串，直接返回-1
    if (2 * m > n) return -1;
    
    long long aL1 = pow(a1, m, mod1);
    long long aL2 = pow(a2, m, mod2);
    long long h1 = 0, h2 = 0;
    for (int i = 0; i < m; ++i) {
        h1 = (h1 * a1 % mod1 + arr[i]) % mod1;
        h2 = (h2 * a2 % mod2 + arr[i]) % mod2;
        if (h1 < 0) h1 += mod1;
        if (h2 < 0) h2 += mod2;
    }
    
    // 用map存储哈希值对应的最早起始位置（便于检查不重叠）
    map<pll, int> seen;
    seen[{h1, h2}] = 0;  // 第一个子串的起始位置为0
    
    for (int start = 1; start <= n - m; ++start) {
        // 计算当前子串的哈希值（滚动哈希）
        h1 = (h1 * a1 % mod1 - arr[start - 1] * aL1 % mod1 + arr[start + m - 1]) % mod1;
        h2 = (h2 * a2 % mod2 - arr[start - 1] * aL2 % mod2 + arr[start + m - 1]) % mod2;
        if (h1 < 0) h1 += mod1;
        if (h2 < 0) h2 += mod2;
        
        pll current_hash = {h1, h2};
        // 检查当前哈希值是否已存在
        if (seen.count(current_hash)) {
            int prev_start = seen[current_hash];
            // 关键：确保两个子串不重叠（前一个子串的结束位置 <= 当前子串的起始位置）
            if (prev_start + m <= start) {
                return start;  // 找到不重叠的重复子串，返回当前起始位置
            }
        } else {
            // 只存储第一次出现的哈希值（保留更早的起始位置，更容易找到不重叠子串）
            seen[current_hash] = start;
        }
    }
    return -1;  // 没有找到不重叠的重复子串
}
    

int longestDupSubstring(string s) {
    srand((unsigned)time(NULL));
    // 生成两个进制
    int a1 = rand()%75 + 26;
    int a2 = rand()%75 + 26;

    // 生成两个模
    int mod1 = rand()%(INT_MAX - 1000000006) + 1000000006;
    int mod2 = rand()%(INT_MAX - 1000000006) + 1000000006;
    int n = s.size();
    // 先对所有字符进行编码
    vector<int> arr(n);
    for (int i = 0; i < n; ++i) {
        arr[i] = s[i] - 'a';
    }
    // 二分查找的范围是[1, n-1]
    int l = 1, r = n - 1;
    int length = 0, start = -1;
    while (l <= r) {
        int m = l + (r - l + 1) / 2;
        int idx = check(arr, m, a1, a2, mod1, mod2);
        if (idx != -1) {
            // 有重复子串，移动左边界
            l = m + 1;
            length = m;
            start = idx;
        } else {
            // 无重复子串，移动右边界
            r = m - 1;
        }
    }
    return start != -1 ? length : -1;
}


int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int t;
    cin >> t;
    while (t--) {
        string s;
        cin >> s;
        cout << longestDupSubstring(s) << endl;
    }
  
    return 0;
}
