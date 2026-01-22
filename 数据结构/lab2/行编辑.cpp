#include<bits/stdc++.h>
typedef long long LL;
typedef std::pair<int,int> PII;
const int N = 100010;

int n;

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
  
    std::cin>>n;
    while (n--) {
        std::string s;
        std::stack<char> a; 
        std::cin>>s;
        for (char &c: s) {
            if (c != '#') {
                a.push(c);
            } else {
                if (a.size()) {
                    a.pop();
                }
            }
        }
        if (!a.size()) {
            std::cout<<"NULL\n";
        } else {
            std::stack<char> temp;
            while (a.size()) {
                temp.push(a.top());
                a.pop();
            }
            while (temp.size()) {
                std::cout<<temp.top();
                temp.pop();
            }
        }
        std::cout<<"\n";
    }
    return 0;
}
