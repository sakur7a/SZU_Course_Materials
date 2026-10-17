#include<bits/stdc++.h>
typedef long long LL;
typedef std::pair<int,int> PII;
const int N = 100010;

struct Node {
    int value;
    Node* next; 
};


Node* createEmptySLL() {
    Node* head = new Node();
    head->next = nullptr;
    return head;
}

Node* createNode(int value) {
    Node* newNode = new Node();
    newNode->value = value;
    newNode->next = nullptr;
    return newNode;
}

void add(Node* head, int value) {
    Node* current = head;
    while (current->next != nullptr) {
        current = current->next;
    }
    current->next = createNode(value);
}



bool insert(Node* head, int pos, int value) {
    if (pos < 1) {
        return false;
    }

    Node* current = head;
    int cnt = 0;

    while (current != nullptr && cnt < pos - 1) {
        current = current->next;
        cnt++;
    }

    if (current == nullptr) {
        return false;
    }

    Node* newNode = createNode(value);
    newNode->next = current->next;
    current->next = newNode;
    return true;
}

bool remove(Node* head, int pos) {
    if (pos < 1) {
        return false;
    }

    Node* current = head;
    int cnt = 0;

    while (current != nullptr && cnt < pos - 1) {
        current = current->next;
        cnt++;
    }

    if (current == nullptr || current->next == nullptr) {
        return false;
    }

    Node* temp = current->next;
    current->next = temp->next;

    delete temp;
    return true;
}

int* find(Node* head, int pos) {
    if (pos < 1) {
        return nullptr;
    }

    Node* current = head;
    int cnt = 0;

    while (current != nullptr && cnt < pos) {
        current = current->next;
        cnt++;
    }

    if (current == nullptr) {
        return nullptr;
    }
    return &current->value;
}

void print(Node* head) {
    Node* current = head->next;
    while (current != nullptr) {
        std::cout<<current->value<<" ";
        current = current->next;
    }
    std::cout<<"\n";
}

void freeList(Node* head) {
    Node* current = head;
    while (current != nullptr) {
        Node* temp = current;
        current = current->next;
        delete temp;
    }
}


Node* LL_merge(Node* a, Node* b) {
    Node* head = createEmptySLL();
    Node* current = head;  
    Node* i = a->next;     
    Node* j = b->next;     
    
    // 合并两个有序链表
    while (i != nullptr && j != nullptr) {
        if (i->value <= j->value) {
            current->next = i;
            i = i->next;
        } else {
            current->next = j;
            j = j->next;
        }
        current = current->next;
    }
    
    // 处理剩余节点(直接指向剩余的链表)
    if (i != nullptr) {
        current->next = i;
    } else {
        current->next = j;
    }

    a->next = nullptr;
    b->next = nullptr;

    return head;
}


int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    Node* list1 = createEmptySLL();
    Node* list2 = createEmptySLL();

    int n, m, x;
    std::cin>>n;

    while (n--) {
        std::cin>>x;
        add(list1, x);
    }

    std::cin>>m;

    while (m--) {
        std::cin>>x;
        add(list2, x);
    }
    
    Node* head = LL_merge(list1, list2);

    print(head);

    freeList(head);
    freeList(list1);
    freeList(list2);
  
    return 0;
}