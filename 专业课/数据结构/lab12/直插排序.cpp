#include <iostream>

using namespace std;
const int N = 100010;
int arr[N];

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n;
    cin >> n;
    
    for (int k = 0; k < n; k++) {
        cin >> arr[k];
    }

    // 直接插入排序核心逻辑
    // 从数组的第二个元素开始（下标1），因为第一个元素默认已是有序序列
    for (int i = 1; i < n; i++) {
        int temp = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > temp) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = temp;

        for (int k = 0; k < n; k++) {
            cout << arr[k];
            if (k < n - 1) {
                cout << " ";
            }
        }
        cout << endl;
    }

    return 0;
}