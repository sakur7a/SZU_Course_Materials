#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

struct SparseEntry {
    int idx;
    double val;
};

struct SparseUser {
    std::vector<SparseEntry> entries;  // 按idx升序存储
    double norm = 0.0;
};

double sparse_cosine_similarity(const SparseUser& a, const SparseUser& b) {
    double dot = 0.0;
    size_t i = 0;
    size_t j = 0;
    while (i < a.entries.size() && j < b.entries.size()) {
        if (a.entries[i].idx == b.entries[j].idx) {
            dot += a.entries[i].val * b.entries[j].val;
            ++i;
            ++j;
        } else if (a.entries[i].idx < b.entries[j].idx) {
            ++i;
        } else {
            ++j;
        }
    }
    return dot / (a.norm * b.norm);
}

std::vector<SparseUser> generate_sparse_users(int n, int d, int s, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> val_dist(0.0, 1.0);
    std::uniform_int_distribution<int> idx_dist(0, d - 1);

    std::vector<SparseUser> users(n);
    for (int i = 0; i < n; ++i) {
        std::unordered_set<int> picked_set;
        picked_set.reserve(static_cast<size_t>(s) * 2);
        while (static_cast<int>(picked_set.size()) < s) {
            picked_set.insert(idx_dist(rng));
        }
        std::vector<int> picked(picked_set.begin(), picked_set.end());
        std::sort(picked.begin(), picked.end());

        users[i].entries.reserve(s);
        double sq_sum = 0.0;
        for (int idx : picked) {
            double v = val_dist(rng);
            users[i].entries.push_back({idx, v});
            sq_sum += v * v;
        }
        users[i].norm = std::sqrt(sq_sum);
    }
    return users;
}

std::vector<std::pair<double, int>> build_similarity_list_sparse(const std::vector<SparseUser>& users,
                                                                  int target_idx) {
    const SparseUser& target = users[target_idx];
    std::vector<std::pair<double, int>> sims;
    sims.reserve(users.size() - 1);

    for (int i = 0; i < static_cast<int>(users.size()); ++i) {
        if (i == target_idx) {
            continue;
        }
        double sim = sparse_cosine_similarity(target, users[i]);
        sims.push_back({sim, i});
    }
    return sims;
}

double run_baseline_full_sort(const std::vector<SparseUser>& users, int target_idx, int k) {
    auto t0 = std::chrono::high_resolution_clock::now();

    auto sims = build_similarity_list_sparse(users, target_idx);
    std::sort(sims.begin(), sims.end(),
              [](const std::pair<double, int>& lhs, const std::pair<double, int>& rhs) {
                  return lhs.first > rhs.first;
              });

    volatile double sink = 0.0;
    for (int i = 0; i < k && i < static_cast<int>(sims.size()); ++i) {
        sink += sims[i].first;
    }
    (void)sink;

    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = t1 - t0;
    return elapsed.count();
}

double run_optimized_topk_heap(const std::vector<SparseUser>& users, int target_idx, int k) {
    auto t0 = std::chrono::high_resolution_clock::now();

    if (k <= 0) {
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = t1 - t0;
        return elapsed.count();
    }

    const SparseUser& target = users[target_idx];
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
        double sim = sparse_cosine_similarity(target, users[i]);
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

double compute_mean(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
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

enum class TheoryModel {
    Baseline,
    Optimized,
};

double complexity_scale(TheoryModel model, int n, int d, int s, int k) {
    // 稀疏场景下，单次相似度由O(d)近似变为O(s)量级（双指针交集）。
    double sim_part = static_cast<double>(n) * static_cast<double>(s);
    if (model == TheoryModel::Baseline) {
        return sim_part + static_cast<double>(n) * std::log(static_cast<double>(n)) + k;
    }
    double safe_k = static_cast<double>(std::max(2, k));
    return sim_part + static_cast<double>(n) * std::log(safe_k) + k;
}

int main() {
    const int d = 10000;
    const int n = 30000;
    const int k = 10;
    const int samples = 12;
    const int warmup_runs = 1;
    const double ci95_z = 1.96;
    const int baseline_s = 40;

    const std::vector<int> s_values = {5, 10, 20, 40, 80, 160, 320};

    std::mt19937_64 rng(20260320ULL);

    struct Algo {
        std::string name;
        TheoryModel model;
        std::function<double(const std::vector<SparseUser>&, int, int)> runner;
    };

    const std::vector<Algo> algos = {
        {"baseline_full_sort", TheoryModel::Baseline, run_baseline_full_sort},
        {"optimized_topk_heap", TheoryModel::Optimized, run_optimized_topk_heap},
    };

    std::ofstream csv("sparse_timing_results.csv");
    if (!csv.is_open()) {
        std::cerr << "无法创建 sparse_timing_results.csv" << std::endl;
        return 1;
    }
    csv << "n,d,k,s,algorithm,avg_ms,stddev_ms,ci95_ms,theory_ms,speedup_vs_baseline\n";

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "开始稀疏实验: n=" << n << ", d=" << d << ", k=" << k
              << ", samples=" << samples << ", warmup_runs=" << warmup_runs << std::endl;

    std::map<std::string, double> baseline_time_at_baseline_s;
    struct Row {
        int s;
        std::string algo;
        double avg_ms;
        double stddev_ms;
        double ci95_ms;
    };
    std::vector<Row> rows;

    for (int s : s_values) {
        std::vector<std::vector<double>> measures(algos.size());

        for (int rep = 0; rep < samples; ++rep) {
            auto users = generate_sparse_users(n, d, s, rng);
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

            if (s == baseline_s) {
                baseline_time_at_baseline_s[algos[a].name] = avg;
            }

            rows.push_back({s, algos[a].name, avg, stddev, ci95});
            std::cout << "s=" << std::setw(4) << s << "  algo=" << std::setw(20) << algos[a].name
                      << "  avg=" << avg << " ms"
                      << "  stddev=" << stddev << " ms"
                      << "  ci95=" << ci95 << " ms" << std::endl;
        }
    }

    std::map<std::string, TheoryModel> model_by_algo;
    for (const auto& algo : algos) {
        model_by_algo[algo.name] = algo.model;
    }

    for (const auto& row : rows) {
        double baseline_at_s0 = baseline_time_at_baseline_s[row.algo];
        double scale = complexity_scale(model_by_algo[row.algo], n, d, row.s, k);
        double scale_s0 = complexity_scale(model_by_algo[row.algo], n, d, baseline_s, k);
        double theory_ms = baseline_at_s0 * (scale / scale_s0);

        double baseline_full_sort_avg = 0.0;
        for (const auto& candidate : rows) {
            if (candidate.s == row.s && candidate.algo == "baseline_full_sort") {
                baseline_full_sort_avg = candidate.avg_ms;
                break;
            }
        }
        double speedup = baseline_full_sort_avg / row.avg_ms;

        csv << n << ',' << d << ',' << k << ',' << row.s << ',' << row.algo << ',' << row.avg_ms << ','
            << row.stddev_ms << ',' << row.ci95_ms << ',' << theory_ms << ',' << speedup << '\n';
    }

    std::cout << "\n实验完成，结果已写入 sparse_timing_results.csv" << std::endl;
    return 0;
}
