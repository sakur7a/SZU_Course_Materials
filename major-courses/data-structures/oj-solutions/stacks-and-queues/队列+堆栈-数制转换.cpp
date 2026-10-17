#include <bits/stdc++.h>
using namespace std;
typedef long long LL;

char getChar(int val) {
    if (0 <= val && val <= 9) {
        return (char)(val + '0');
    } else {
        return (char)(val - 10 + 'A');
    }
}

void solve() {
    double n;
    int k;
    std::cin >> n >> k;
    LL int_part = (LL)n;
    double frac_part = n - (double)int_part;

    stack<char> stk;
    if (int_part == 0) {
        stk.push('0');
    } else {
        while (int_part) {
            stk.push(getChar(int_part % k));
            int_part /= k;
        }
    }

    queue<char> q;
    double temp_frac = frac_part;
    for (int i = 0; i < 3; i++) {
        temp_frac *= k;
        int digit = (int)temp_frac;
        q.push(getChar(digit));
        temp_frac -= digit;
    }

    while (stk.size()) {
        std::cout << stk.top();
        stk.pop();
    }
    std::cout << ".";

    while (q.size()) {
        std::cout << q.front();
        q.pop();
    }
    std::cout << "\n";


}

int main() {
    int t;
    std::cin >> t;
    while (t--) {
        solve();
    }



    return 0;
}