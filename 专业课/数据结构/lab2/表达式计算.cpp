#include<bits/stdc++.h>

int t;
std::string s;
std::stack<char> op;
std::stack<double> num;
std::unordered_map<char, int> cmp{{'+', 1}, {'-', 1}, {'*', 2}, {'/', 2}};

void eval() {
    double second = num.top();
    num.pop();
    double first = num.top();
    num.pop();

    char p = op.top();
    op.pop();

    double res = 0;
    if (p == '+') {
        res = first + second;
    }
    if (p == '-') {
        res = first - second;
    }
    if (p == '*') {
        res = first * second;
    }
    if (p == '/') {
        res = first / second;
    }

    num.push(res);
}


int main()
{
    std::cin>>t;
    while (t--) {
        while (op.size()) {
            op.pop();
        }
        while (num.size()) {
            num.pop();
        }

        std::cin>>s;
        s = s.substr(0, s.size() - 1);
        for (int i = 0; i < s.size(); i++) {
            char c = s[i];
            if (isdigit(c) || c == '.') {
                bool st = false; //标记是否有小数部分
                double x = 0;
                int index = 0;
                int j = i;

                while (j < s.size() && (isdigit(s[j]) || s[j] == '.')) {
                    if (s[j] == '.') {
                        j++;
                        st = true;
                    }
                    if (!st) {
                        x = x * 10 + (s[j] - '0');
                    } else {
                        index++;
                        x += (s[j] - '0') / pow(10, index);
                    }
                    j++;
                }
                num.push(x);
                i = j - 1;
            } else if (c == '(') {
                op.push(c);
            } else if (c == ')') {
                while (op.top() != '(') {
                    eval();
                }
                op.pop();
            } else {
                if (op.size() && cmp[op.top()] > cmp[c]) {
                    eval();
                }
                op.push(c);
            }
        }

        while (op.size()) {
            eval();
        }
        std::cout<<std::fixed<<std::setprecision(4)<<num.top()<<"\n";
    }
    return 0;
}