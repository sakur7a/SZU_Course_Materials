#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace std;

// 定义国家结构体，存储原始数据和计算后的比率
struct Country {
    int id;
    int gold;
    int total;
    int pop;
    double g_ratio; // 人均金牌
    double t_ratio; // 人均奖牌
};

int main() {
    // 优化输入输出效率
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int N, M;
    if (cin >> N >> M) {
        vector<Country> countries(N);
        for (int i = 0; i < N; i++) {
            countries[i].id = i;
            cin >> countries[i].gold >> countries[i].total >> countries[i].pop;
            
            // 计算人均数据，注意转换为浮点数运算
            // 题目保证数据范围在 [0, 1000]，通常人口不为0，但若为0则比率为0
            if (countries[i].pop == 0) {
                countries[i].g_ratio = 0;
                countries[i].t_ratio = 0;
            } else {
                countries[i].g_ratio = (double)countries[i].gold / countries[i].pop;
                countries[i].t_ratio = (double)countries[i].total / countries[i].pop;
            }
        }

        // ranks[id][m] 存储 ID 为 id 的国家在第 m 种方式下的排名
        // m: 0-金牌, 1-奖牌, 2-人均金, 3-人均奖
        vector<vector<int>> ranks(N, vector<int>(4));

        // 对 4 种方式分别进行排序和排名计算
        for (int m = 0; m < 4; m++) {
            // 创建一个索引数组用于排序
            vector<int> p(N);
            for (int i = 0; i < N; i++) p[i] = i;

            // 根据当前方式 m 对索引数组进行降序排序
            sort(p.begin(), p.end(), [&](int a, int b) {
                double va, vb;
                switch (m) {
                    case 0: va = countries[a].gold; vb = countries[b].gold; break;
                    case 1: va = countries[a].total; vb = countries[b].total; break;
                    case 2: va = countries[a].g_ratio; vb = countries[b].g_ratio; break;
                    case 3: va = countries[a].t_ratio; vb = countries[b].t_ratio; break;
                }
                return va > vb;
            });

            // 根据排序结果填写 ranks 表
            for (int i = 0; i < N; i++) {
                int curr_id = p[i];
                if (i == 0) {
                    ranks[curr_id][m] = 1;
                } else {
                    int prev_id = p[i - 1];
                    double v_curr, v_prev;
                    // 获取当前和前一个的值进行比较
                    switch (m) {
                        case 0: v_curr = countries[curr_id].gold; v_prev = countries[prev_id].gold; break;
                        case 1: v_curr = countries[curr_id].total; v_prev = countries[prev_id].total; break;
                        case 2: v_curr = countries[curr_id].g_ratio; v_prev = countries[prev_id].g_ratio; break;
                        case 3: v_curr = countries[curr_id].t_ratio; v_prev = countries[prev_id].t_ratio; break;
                    }

                    // 如果分数相同，排名并列；否则排名为 i+1 (即绝对排名)
                    // 浮点数比较使用极小误差 1e-10 保证稳健性
                    if (abs(v_curr - v_prev) < 1e-10) {
                        ranks[curr_id][m] = ranks[prev_id][m];
                    } else {
                        ranks[curr_id][m] = i + 1;
                    }
                }
            }
        }

        // 处理查询
        vector<int> queries(M);
        for (int i = 0; i < M; i++) {
            cin >> queries[i];
        }

        for (int i = 0; i < M; i++) {
            int id = queries[i];
            int best_rank = N + 1; // 初始化一个比所有可能排名都大的值
            int best_method = -1;

            // 遍历 4 种方式，寻找最优排名
            // 由于 m 从 0 到 3 遍历，且只有遇到更小排名(<)时才更新
            // 所以如果有并列的最优排名，保留的是 m 最小的那个，符合题目要求
            for (int m = 0; m < 4; m++) {
                if (ranks[id][m] < best_rank) {
                    best_rank = ranks[id][m];
                    best_method = m + 1; // 题目要求的输出编号是 1-4
                }
            }

            cout << best_rank << ":" << best_method;
            if (i < M - 1) cout << " ";
        }
        cout << endl;
    }

    return 0;
}