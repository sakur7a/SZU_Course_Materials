#include <bits/stdc++.h>
using namespace std;
typedef long long LL;
typedef std::pair<int, int> PII;
typedef std::pair<LL, LL> PLL;
const int N = 100010, MOD = 11;

int t, n, m;

struct Node {
    int data;
    Node* next;
    Node(int val) : data(val), next(NULL) {}
};

Node* hashTable[MOD];

// 哈希函数
int hashFunc(int key) {
    return key % MOD;
}

void insert(int key) {
    int idx = hashFunc(key);
    Node* newNode = new Node(key);
    
    if (hashTable[idx] == NULL) {
        hashTable[idx] = newNode;
    } else {
        Node* curr = hashTable[idx];
        while (curr->next != NULL) {
            curr = curr->next;
        }
        curr->next = newNode;
    }

}

void searchAndProcess(int key) {
    int idx = hashFunc(key);
    Node* curr = hashTable[idx];
    int count = 0;
    bool found = false;
    
    while (curr != NULL) {
        count++; 
        if (curr->data == key) {
            found = true;
            break;
        }
        curr = curr->next;
    }
    
    if (found) {
        cout << idx << " " << count << endl;
    } else {
        cout << "error" << endl;
        insert(key);
    }
}

int main() {
    for (int i = 0; i < MOD; ++i) {
        hashTable[i] = NULL;
    }
    
    cin >> n;
    for (int i = 0; i < n; ++i) {
        int val;
        cin >> val;
        insert(val);
    }
    
    int t;
    cin >> t;
    while (t--) {
        int key;
        cin >> key;
        searchAndProcess(key);
    }
    
    
    return 0;
}