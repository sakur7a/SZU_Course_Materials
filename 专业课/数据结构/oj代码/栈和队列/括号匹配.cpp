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
        bool st = true;
        for (char &c: s) {
            if (c == '(' || c == '[' || c == '{') {
                a.push(c);
            }
            if (c == ')' || c == ']' || c == '}') {
                if (a.size()) {
                    if (c == ')' && a.top() != '(' || c == ']' && a.top() != '[' || c == '}' && a.top() != '{') {
                        st = false;
                        break;
                    } else {
                        a.pop();
                    }
                } else {
                    st = false;
                    break;
                }
            }
        }
        if (a.size()) {
            st = false;
        }
        if (st) {
            std::cout<<"ok\n";
        } else {
            std::cout<<"error\n";
        }

    }
    return 0;
}
