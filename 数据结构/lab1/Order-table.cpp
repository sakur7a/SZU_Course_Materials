#include <bits/stdc++.h>
using namespace std;
const int N = 1010;


class SeqList {
private:
    int data[N];
    int length;

public:
    SeqList() {
        length = 0;
    }

    void create(int n) {
        length = n;
        for (int i = 0; i < n; i++) {
            std::cin >> data[i];
        }
        print();
    }

    void insert(int pos, int val) {
        if (pos < 1 || pos > length + 1 || length >= N) {
            std::cout << "error\n";
            return;
        }

        for (int i = length; i >= pos; i--) {
            data[i] = data[i - 1];
        }
        data[pos - 1] = val;
        length++;
        print();
    }

    void remove(int pos) {
        if (pos < 1 || pos > length) {
            std::cout << "error\n";
            return;
        } 
        for (int i = pos - 1; i < length - 1; i++) {
            data[i] = data[i + 1];
        }
        length--;
        print();
    }


    void find(int pos) {
        if (pos < 1 || pos > length) {
            std::cout << "error\n";
            return;
        } 
        std::cout << data[pos - 1];
    }


    void multiinsert(int i, int n, int item[]) {
        if (length + n > N || i < 1 || i > length + 1) {
            std::cout << "errpr\n";
            return;
        }

        for (int k =  length - 1; k >= i - 1; k--) {
            data[k + n] = data[k];
        } 

        for (int k = 0; k < n; k++) {
            data[i + k - 1] = item[k];
        }
        length += n;
        print();
    }

    void multidel(int i, int n) {
        if (i < 1 || i > length || i - 1 + n > length) {
            std::cout << "error\n";
            return;
        }

        for (int k = i + n - 1; k < length; k++) {
            data[k - n] = data[k];
        }
        length -= n;
        print();
    }

    void print() {
        std::cout << length << " ";
        for (int i = 0; i < length; i++) {
            std::cout << data[i] << " ";
        }
        std::cout << "\n";
    }
};


int main() {
    int n, pos, val;
    SeqList list;
    std::cin >> n;
    
    list.create(n);

    int i, k;
    std::cin >> i >> k;
    
    int *temp = new int[k];
    for (int j = 0; j < k; j++) {
        std::cin >> temp[j];
    }
    list.multiinsert(i, k, temp);
    delete[] temp;

    std::cin >> i >> k;
    list.multidel(i, k);


    return 0;
}