#include<bits/stdc++.h>
typedef long long LL;
typedef std::pair<int,int> PII;
const int N = 100010;

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    
    int n;
    std::cin >> n;
    
    std::vector<char> types(n);
    std::vector<int> times(n);
    
    for (int i = 0; i < n; i++) {
        std::cin >> types[i];
    }
    
    for (int i = 0; i < n; i++) {
        std::cin >> times[i];
    }
    
    std::queue<int> queueA, queueB, queueC;
    
    for (int i = 0; i < n; i++) {
        if (types[i] == 'A') {
            queueA.push(times[i]);
        } else if (types[i] == 'B') {
            queueB.push(times[i]);
        } else if (types[i] == 'C') {
            queueC.push(times[i]);
        }
    }
    
    int totalA = 0, countA = 0;
    int totalB = 0, countB = 0;
    int totalC = 0, countC = 0;
    

    int currentTime = 0;
    while (!queueA.empty()) {
        int serviceTime = queueA.front();
        queueA.pop();
        currentTime += serviceTime;
        totalA += currentTime;
        countA++;
    }
    

    currentTime = 0;
    while (!queueB.empty()) {
        int serviceTime = queueB.front();
        queueB.pop();
        currentTime += serviceTime;
        totalB += currentTime;
        countB++;
    }

    currentTime = 0;
    while (!queueC.empty()) {
        int serviceTime = queueC.front();
        queueC.pop();
        currentTime += serviceTime;
        totalC += currentTime;
        countC++;
    }
    
    if (countA > 0) {
        std::cout << totalA / countA << "\n";
    } else {
        std::cout << "0\n";
    }
    
    if (countB > 0) {
        std::cout << totalB / countB << "\n";
    } else {
        std::cout << "0\n";
    }
    
    if (countC > 0) {
        std::cout << totalC / countC << "\n";
    } else {
        std::cout << "0\n";
    }
    
    return 0;
}