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
            a.push(c);
        }
        while (a.size()) {
            std::cout<<a.top();
            a.pop();
        }
        std::cout<<"\n";
        ;
    }
    return 0;
}
