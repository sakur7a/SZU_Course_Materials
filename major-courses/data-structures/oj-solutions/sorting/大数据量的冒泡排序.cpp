#include <bits/stdc++.h>
using namespace std;
typedef long long LL;
typedef std::pair<int, int> PII;
typedef std::pair<LL, LL> PLL;
const int N = 5010;

int t, n, m;
int a[N], temp[N];
LL res;

void merge_sort(int l, int r) {
    if (l >= r) {
        return;
    }
    int mid = (l + r) >> 1;
    merge_sort(l, mid), merge_sort(mid + 1, r);
    
    int i = l, j = mid + 1;
    int k = 0;
    while (i <= mid && j <= r) {
        if (a[i] <= a[j]) {
            temp[k++] = a[i++];
        } else {
            temp[k++] = a[j++];
            res += mid - i + 1;
        }
    }

    while (i <= mid) {
        temp[k++] = a[i++];
    }

    while (j <= r) {
        temp[k++] = a[j++];
    }

    for (int i = l, j = 0; j < k; i++, j++) {
        a[i] = temp[j];
    }
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    while (std::cin >> n) {
        for (int i = 0; i < n; i++) {
            std::cin >> a[i];
        }
        res = 0;
        merge_sort(0, n - 1);
        std::cout << res << "\n";
    }
  
    return 0;
}
