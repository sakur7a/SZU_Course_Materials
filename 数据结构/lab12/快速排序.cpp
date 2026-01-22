#include <iostream>

using namespace std;
int n;

// 打印数组函数
void printArr(int* arr) {
    for (int i = 0; i < n; i++) {
        cout << arr[i] << (i == n - 1 ? "" : " ");
    }
    cout << endl;
}

int partition(int arr[], int low, int high) {
    int pivot = arr[low];
    int i = low, j = high;
    while (i < j) {
        while (i < j && arr[j] >= pivot) {
            j--;
        }
        if (i < j) {
            arr[i] = arr[j];
        }

        while (i < j && arr[i] <= pivot) {
            i++;
        }
        if (i < j) {
            arr[j] = arr[i];
        }
    }
    arr[i] = pivot;
    return i;
}

void qsort(int arr[], int low, int high) {
    if (low < high) {
        int pivotPos = partition(arr, low, high);
        printArr(arr);
        qsort(arr, low, pivotPos - 1);
        qsort(arr, pivotPos + 1, high);
    }
}

void solve() {

    // 读取数组长度
    cin >> n;
    
    // 动态分配内存
    int* arr = new int[n];
    for (int i = 0; i < n; i++) {
        cin >> arr[i];
    }
    
    // 开始快排
    qsort(arr, 0, n - 1);
    
    // 释放内存
    delete[] arr;
}

int main() {
    // 优化 I/O 性能
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int t;
    if (cin >> t) {
        while (t--) {
            solve();
            // 不同测试数据间用空行分隔
            // 如果不是最后一组数据，输出空行
            if (t > 0) cout << endl;
        }
    }
    return 0;
}