#include <iostream>
#include <string>
#include <stack>

using namespace std;

int main() {
    string s;
    // 读取代表队列的字符串
    if (!(cin >> s)) return 0;

    // 题目规定：首先出现的字符代表小男孩
    char boyChar = s[0];
    
    // 使用栈来存储小男孩的编号（索引）
    stack<int> stk;

    for (int i = 0; i < s.length(); ++i) {
        if (s[i] == boyChar) {
            // 如果是男孩，将其编号入栈
            stk.push(i);
        } else {
            // 如果是女孩，她会与当前栈顶（即离她最近）的男孩配对
            if (!stk.empty()) {
                int boyIndex = stk.top();
                stk.pop();
                // 按照“小男孩编号 小女孩编号”格式输出
                // 因为我们是从左往右遍历，遇到女孩的先后顺序就是女孩编号从小到大的顺序
                cout << boyIndex << " " << i << endl;
            }
        }
    }

    return 0;
}