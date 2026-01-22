#include<bits/stdc++.h>
typedef long long LL;
typedef std::pair<int,int> PII;
const int N = 100010;

int n;
std::string op, url, current;
std::stack<std::string> front, rear;

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    current = "https://www-acm-org.webvpn.szu.edu.cn/";
    while (1) {
        std::cin>>op;
        if (op == "VISIT") {
            std::cin>>url;
            rear.push(current);
            current = url;
            while (front.size()) {
                front.pop();
            }
            std::cout<<current<<"\n";
        } else if (op == "BACK") {
            if (!rear.size()) {
                std::cout<<"Ignored\n";
            } else {
                front.push(current);
                current = rear.top();
                rear.pop();
                std::cout<<current<<"\n";
            }
        } else if (op == "FORWARD") {
            if (!front.size()) {
                std::cout<<"Ignored\n";
            } else {
                rear.push(current);
                current = front.top();
                front.pop();
                std::cout<<current<<"\n";
            }
        } else {
            break;
        }
    } 
    return 0;
}
