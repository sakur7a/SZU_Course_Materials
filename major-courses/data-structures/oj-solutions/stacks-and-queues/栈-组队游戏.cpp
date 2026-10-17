#include<bits/stdc++.h>
typedef long long LL;
typedef std::pair<int, int> PII;
const int N = 100010;

std::string s;
std::unordered_map<char, char> mp;
char first, second;
std::stack<PII> stk;
std::vector<PII> ans;


void solve() {
    std::cin>>s;

    for (char &c : s) {
        if (!first) {
            first = c;
            mp[first] = '(';
        } else if (!second && c != first) {
            second = c;
            mp[second] = ')';
            break;
        }
    }

    for (char &c : s) {
        c = mp[c];
    }

    for (int i = 0; i < s.size(); i++) {
        if (s[i] == '(') {
            stk.push({s[i], i});
        } else if (s[i] == ')' && stk.size()) {
            ans.push_back({stk.top().second, i});
            stk.pop();
        }
    }

    std::sort(ans.begin(), ans.end(), [&](PII &a, PII &b) {
        return a.second < b.second; 
        }
    );

    for (auto &p : ans) {
        std::cout << p.first << " " << p.second << "\n";
    }
    
}



int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    solve();
    return 0;
}
