#include <iostream>
using namespace std;

struct Node {
    int coef;
    int exp;
    Node* next;
    Node(int c, int e) : coef(c), exp(e), next(nullptr) {}
};

void insert(Node* head, int c, int e) {
    if (c == 0) {
        return;
    }

    Node* cur = head;
    while (cur->next && cur->next->exp < e) {
        cur = cur->next;
    }

    if (cur->next && cur->next->exp == e) {
        cur->next->coef += c;
        if (cur->next->coef == 0) {
            Node* temp = cur->next;
            cur->next = temp->next;
            delete temp; 
        }
    } else {
        Node* newNode = new Node(c, e);
        newNode->next = cur->next;
        cur->next = newNode;
    }
}

void print(Node* head) {
    Node* p = head->next;
    bool first = true;
    while (p) {
        if (p->coef == 0) {
            p = p->next;
            continue;
        } // just for robustness'

        if (!first) {
            std::cout << " + ";
        }

        if (p->coef < 0) {
            std::cout << "(" << p->coef << ")";
        } else {
            std::cout << p->coef;
        }

        if (p->exp != 0) {
            std::cout << "x^";
            if (p->exp < 0) {cout << "(" << p->exp << ")";}
            else cout << p->exp;
        }

        first = false;
        p = p->next;
    }
    std::cout << "\n";
}

Node* add(Node* p1, Node* p2) {
    Node* res = new Node(0, 0);
    Node *a = p1->next, *b = p2->next;

    while (a || b) {
        if (a && (!b || a->exp < b->exp)) {
            insert(res, a->coef, a->exp);
            a = a->next;
        } else if (b && (!a || a->exp > b->exp)) {
            insert(res, b->coef, b->exp);
            b = b->next;
        } else {
            insert(res, a->coef + b->coef, a->exp);
            a = a->next, b = b->next;
        }
    }
    return res;
}


int n, t, m, c, e;

int main() {

    std::cin >> t;
    while (t--) {
        Node* a = new Node(0, 0);
        Node* b = new Node(0, 0);
        std::cin >> n;
        for (int i = 0; i < n; i++) {
            std::cin >> c >> e;
            insert(a, c, e);
        }
        print(a);

        std::cin >> m;
        for (int i = 0; i < m; i++) {
            std::cin >> c >> e;
            insert(b, c, e);
        }
        print(b);

        Node* res = add(a, b);
        print(res);
    }
    return 0;
}