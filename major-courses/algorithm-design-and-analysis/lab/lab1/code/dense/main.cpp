#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <map>
#include <queue>
#include <string>
#include <utility>
#include <vector>

struct User {
    std::vector<double> vec;
    double norm = 0.0;
};

double cosine_similarity(const User& a, const User& b) {
    double dot = 0.0;
    for (size_t i = 0; i < a.vec.size(); ++i) {
        dot += a.vec[i] * b.vec[i];
    }
    return dot / (a.norm * b.norm);
}

std::vector<User> generate_users(int n, int d, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::vector<User> users(n);
    for (int i = 0; i < n; ++i) {
        users[i].vec.resize(d);
        double sq_sum = 0.0;
        for (int j = 0; j < d; ++j) {
            double v = dist(rng);
            users[i].vec[j] = v;
            sq_sum += v * v;
        }
        users[i].norm = std::sqrt(sq_sum);
    }
    return users;
}

std::vector<std::pair<double, int>> build_similarity_list(const std::vector<User>& users, int target_idx) {
    const User& target = users[target_idx];
    std::vector<std::pair<double, int>> sims;
    sims.reserve(users.size() - 1);

    for (int i = 0; i < static_cast<int>(users.size()); ++i) {
        if (i == target_idx) {
            continue;
        }
        double sim = cosine_similarity(target, users[i]);
        sims.push_back({sim, i});
    }
    return sims;
}

void sort_descending(std::vector<std::pair<double, int>>& sims) {
    std::sort(
        sims.begin(), sims.end(),
        [](const std::pair<double, int>& lhs, const std::pair<double, int>& rhs) {
            return lhs.first > rhs.first;
        });
}

// 基于指定排序函数执行“算相似度 + 全量排序 + 取前k”的完整流程，并返回耗时(毫秒)
double run_recommendation_full_sort(const std::vector<User>& users, int target_idx, int k) {
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<std::pair<double, int>> sims = build_similarity_list(users, target_idx);
    sort_descending(sims);

    volatile double sink = 0.0;
    for (int i = 0; i < k && i < static_cast<int>(sims.size()); ++i) {
        sink += sims[i].first;
    }
    (void)sink;

    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = t1 - t0;
    return elapsed.count();
}

// 优化版：用大小为k的最小堆维护Top-k，避免对全部相似度做全量排序。
double run_recommendation_topk_heap(const std::vector<User>& users, int target_idx, int k) {
    auto t0 = std::chrono::high_resolution_clock::now();

    if (k <= 0) {
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = t1 - t0;
        return elapsed.count();
    }

    const User& target = users[target_idx];
    auto cmp_min_heap = [](const std::pair<double, int>& lhs, const std::pair<double, int>& rhs) {
        return lhs.first > rhs.first;
    };
    std::priority_queue<std::pair<double, int>,
                        std::vector<std::pair<double, int>>,
                        decltype(cmp_min_heap)>
        topk(cmp_min_heap);

    for (int i = 0; i < static_cast<int>(users.size()); ++i) {
        if (i == target_idx) {
            continue;
        }
        double sim = cosine_similarity(target, users[i]);
        if (static_cast<int>(topk.size()) < k) {
            topk.push({sim, i});
        } else if (sim > topk.top().first) {
            topk.pop();
            topk.push({sim, i});
        }
    }

    volatile double sink = 0.0;
    while (!topk.empty()) {
        sink += topk.top().first;
        topk.pop();
    }
    (void)sink;

    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = t1 - t0;
    return elapsed.count();
}

enum class ComplexityModel {
    FullSort,
    TopKHeap,
};

double complexity_scale(ComplexityModel model, int n, int k, int dim) {
    const double nd = static_cast<double>(n) * static_cast<double>(dim);
    const double nk = static_cast<double>(k);
    if (model == ComplexityModel::FullSort) {
        return nd + static_cast<double>(n) * std::log(static_cast<double>(n)) + nk;
    }
    const double safe_k = static_cast<double>(std::max(2, k));
    return nd + static_cast<double>(n) * std::log(safe_k) + nk;
}

double compute_mean(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    return sum / static_cast<double>(values.size());
}

double compute_sample_stddev(const std::vector<double>& values, double mean) {
    if (values.size() < 2) {
        return 0.0;
    }
    double accum = 0.0;
    for (double v : values) {
        double diff = v - mean;
        accum += diff * diff;
    }
    return std::sqrt(accum / static_cast<double>(values.size() - 1));
}

int main() {
    const int dim = 50;
    const int samples = 20;
    const int warmup_runs = 2;
    const int baseline_n = 100000;
    const double ci95_z = 1.96;

    // 横坐标均匀取值
    const std::vector<int> n_values = {
        5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 60000, 80000, 100000
    };
    const std::vector<int> k_values = {10, 50, 100};

    std::mt19937_64 rng(20260320ULL);

    struct Algo {
        std::string name;
        std::function<double(const std::vector<User>&, int, int)> runner;
        ComplexityModel model;
    };

    const std::vector<Algo> algos = {
        {"baseline_full_sort", run_recommendation_full_sort, ComplexityModel::FullSort},
        {"optimized_topk_heap", run_recommendation_topk_heap, ComplexityModel::TopKHeap},
    };

    std::ofstream csv("timing_results.csv");
    if (!csv.is_open()) {
        std::cerr << "无法创建 timing_results.csv" << std::endl;
        return 1;
    }
    csv << "k,n,algorithm,avg_ms,stddev_ms,ci95_ms,theory_ms,ratio_to_baseline\n";

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "开始实验: dim=" << dim
              << ", samples=" << samples
              << ", warmup_runs=" << warmup_runs
              << ", baseline_n=" << baseline_n << std::endl;

    for (int k : k_values) {
        std::cout << "\n===== k = " << k << " =====" << std::endl;

        std::map<std::string, double> baseline_avg_by_algo;
        struct Row {
            int n;
            std::string algo;
            double avg_ms;
            double stddev_ms;
            double ci95_ms;
        };
        std::vector<Row> rows;

        for (int n : n_values) {
            std::vector<std::vector<double>> measures(algos.size());

            for (int s = 0; s < samples; ++s) {
                auto users = generate_users(n, dim, rng);
                const int target_idx = 0;

                for (size_t a = 0; a < algos.size(); ++a) {
                    for (int w = 0; w < warmup_runs; ++w) {
                        (void)algos[a].runner(users, target_idx, k);
                    }

                    double ms = algos[a].runner(users, target_idx, k);
                    measures[a].push_back(ms);
                }
            }

            for (size_t a = 0; a < algos.size(); ++a) {
                double avg = compute_mean(measures[a]);
                double stddev = compute_sample_stddev(measures[a], avg);
                double ci95 = ci95_z * (stddev / std::sqrt(static_cast<double>(measures[a].size())));

                if (n == baseline_n) {
                    baseline_avg_by_algo[algos[a].name] = avg;
                }

                rows.push_back({n, algos[a].name, avg, stddev, ci95});
                std::cout << "n=" << std::setw(6) << n
                          << "  algo=" << std::setw(12) << algos[a].name
                          << "  avg=" << avg << " ms"
                          << "  stddev=" << stddev << " ms"
                          << "  ci95=" << ci95 << " ms" << std::endl;
            }
        }

        std::map<std::string, ComplexityModel> model_by_algo;
        for (const auto& algo : algos) {
            model_by_algo[algo.name] = algo.model;
        }

        for (const auto& row : rows) {
            double baseline_avg = baseline_avg_by_algo[row.algo];
            ComplexityModel model = model_by_algo[row.algo];
            double current_scale = complexity_scale(model, row.n, k, dim);
            double baseline_scale = complexity_scale(model, baseline_n, k, dim);
            double theory_ms = baseline_avg * (current_scale / baseline_scale);
            double ratio = row.avg_ms / baseline_avg;
            csv << k << ',' << row.n << ',' << row.algo << ','
                << row.avg_ms << ',' << row.stddev_ms << ',' << row.ci95_ms << ','
                << theory_ms << ',' << ratio << '\n';
        }
    }

    std::cout << "\n实验完成，结果已写入 timing_results.csv" << std::endl;
    return 0;
}
