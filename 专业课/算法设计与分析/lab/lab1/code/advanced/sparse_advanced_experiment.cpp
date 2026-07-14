#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

struct SparseEntry {
    int idx;
    float val;
};

struct SparseUser {
    std::vector<SparseEntry> entries;  // sorted by idx
    float norm = 0.0F;
    float inv_norm = 0.0F;
};

struct Posting {
    int user;
    float val;
};

struct QueryOutput {
    double ms = 0.0;
    std::vector<int> topk_users;
    int candidates = 0;
};

float sparse_dot(const SparseUser& a, const SparseUser& b) {
    float dot = 0.0F;
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
    return dot;
}

std::vector<SparseUser> generate_sparse_users(int n, int d, int s, std::mt19937_64& rng) {
    std::uniform_int_distribution<int> idx_dist(0, d - 1);
    std::uniform_real_distribution<float> val_dist(0.0F, 1.0F);

    std::vector<SparseUser> users(n);
    for (int i = 0; i < n; ++i) {
        std::unordered_set<int> picked;
        picked.reserve(static_cast<size_t>(s) * 2U);
        while (static_cast<int>(picked.size()) < s) {
            picked.insert(idx_dist(rng));
        }

        std::vector<int> idxs(picked.begin(), picked.end());
        std::sort(idxs.begin(), idxs.end());

        users[i].entries.reserve(static_cast<size_t>(s));
        float sq_sum = 0.0F;
        for (int idx : idxs) {
            float v = val_dist(rng);
            users[i].entries.push_back({idx, v});
            sq_sum += v * v;
        }
        users[i].norm = std::sqrt(sq_sum);
        users[i].inv_norm = (users[i].norm > 0.0F) ? (1.0F / users[i].norm) : 0.0F;
    }
    return users;
}

std::vector<std::vector<Posting>> build_inverted_index(const std::vector<SparseUser>& users, int d) {
    std::vector<std::vector<Posting>> inv(static_cast<size_t>(d));
    for (int u = 0; u < static_cast<int>(users.size()); ++u) {
        for (const auto& e : users[u].entries) {
            inv[static_cast<size_t>(e.idx)].push_back({u, e.val});
        }
    }

    for (auto& postings : inv) {
        std::sort(postings.begin(), postings.end(), [](const Posting& a, const Posting& b) {
            return a.val > b.val;
        });
    }
    return inv;
}

QueryOutput query_full_scan_heap(const std::vector<SparseUser>& users, int target_idx, int k) {
    auto t0 = std::chrono::high_resolution_clock::now();

    auto cmp_min_heap = [](const std::pair<float, int>& lhs, const std::pair<float, int>& rhs) {
        return lhs.first > rhs.first;
    };
    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, decltype(cmp_min_heap)> topk(
        cmp_min_heap);

    const SparseUser& target = users[target_idx];
    for (int i = 0; i < static_cast<int>(users.size()); ++i) {
        if (i == target_idx) {
            continue;
        }
        float dot = sparse_dot(target, users[i]);
        float sim = dot * target.inv_norm * users[i].inv_norm;

        if (static_cast<int>(topk.size()) < k) {
            topk.push({sim, i});
        } else if (sim > topk.top().first) {
            topk.pop();
            topk.push({sim, i});
        }
    }

    std::vector<int> result;
    result.reserve(static_cast<size_t>(k));
    while (!topk.empty()) {
        result.push_back(topk.top().second);
        topk.pop();
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = t1 - t0;
    return {elapsed.count(), result, static_cast<int>(users.size()) - 1};
}

QueryOutput query_inverted_exact(const std::vector<SparseUser>& users,
                                 const std::vector<std::vector<Posting>>& inv,
                                 int target_idx,
                                 int k) {
    auto t0 = std::chrono::high_resolution_clock::now();

    const SparseUser& target = users[target_idx];
    std::unordered_map<int, float> dot_map;
    dot_map.reserve(4096);

    for (const auto& e : target.entries) {
        const auto& postings = inv[static_cast<size_t>(e.idx)];
        for (const auto& p : postings) {
            if (p.user == target_idx) {
                continue;
            }
            dot_map[p.user] += e.val * p.val;
        }
    }

    auto cmp_min_heap = [](const std::pair<float, int>& lhs, const std::pair<float, int>& rhs) {
        return lhs.first > rhs.first;
    };
    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, decltype(cmp_min_heap)> topk(
        cmp_min_heap);

    for (const auto& kv : dot_map) {
        int user = kv.first;
        float sim = kv.second * target.inv_norm * users[user].inv_norm;
        if (static_cast<int>(topk.size()) < k) {
            topk.push({sim, user});
        } else if (sim > topk.top().first) {
            topk.pop();
            topk.push({sim, user});
        }
    }

    std::vector<int> result;
    result.reserve(static_cast<size_t>(k));
    while (!topk.empty()) {
        result.push_back(topk.top().second);
        topk.pop();
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = t1 - t0;
    return {elapsed.count(), result, static_cast<int>(dot_map.size())};
}

QueryOutput query_two_stage(const std::vector<SparseUser>& users,
                            const std::vector<std::vector<Posting>>& inv,
                            int target_idx,
                            int k,
                            int posting_cap,
                            int candidate_cap) {
    auto t0 = std::chrono::high_resolution_clock::now();

    const SparseUser& target = users[target_idx];
    std::unordered_map<int, float> approx_dot;
    approx_dot.reserve(static_cast<size_t>(candidate_cap) * 2U);

    for (const auto& e : target.entries) {
        const auto& postings = inv[static_cast<size_t>(e.idx)];
        const int take = std::min(posting_cap, static_cast<int>(postings.size()));
        for (int i = 0; i < take; ++i) {
            int user = postings[static_cast<size_t>(i)].user;
            if (user == target_idx) {
                continue;
            }
            approx_dot[user] += e.val * postings[static_cast<size_t>(i)].val;
        }
    }

    auto cmp_min_heap = [](const std::pair<float, int>& lhs, const std::pair<float, int>& rhs) {
        return lhs.first > rhs.first;
    };
    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, decltype(cmp_min_heap)> cand_heap(
        cmp_min_heap);

    for (const auto& kv : approx_dot) {
        int user = kv.first;
        float approx_sim = kv.second * target.inv_norm * users[user].inv_norm;
        if (static_cast<int>(cand_heap.size()) < candidate_cap) {
            cand_heap.push({approx_sim, user});
        } else if (approx_sim > cand_heap.top().first) {
            cand_heap.pop();
            cand_heap.push({approx_sim, user});
        }
    }

    std::vector<int> candidates;
    candidates.reserve(cand_heap.size());
    while (!cand_heap.empty()) {
        candidates.push_back(cand_heap.top().second);
        cand_heap.pop();
    }

    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, decltype(cmp_min_heap)> topk(
        cmp_min_heap);
    for (int user : candidates) {
        float dot = sparse_dot(target, users[user]);
        float sim = dot * target.inv_norm * users[user].inv_norm;
        if (static_cast<int>(topk.size()) < k) {
            topk.push({sim, user});
        } else if (sim > topk.top().first) {
            topk.pop();
            topk.push({sim, user});
        }
    }

    std::vector<int> result;
    result.reserve(static_cast<size_t>(k));
    while (!topk.empty()) {
        result.push_back(topk.top().second);
        topk.pop();
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = t1 - t0;
    return {elapsed.count(), result, static_cast<int>(candidates.size())};
}

double mean(const std::vector<double>& xs) {
    if (xs.empty()) {
        return 0.0;
    }
    return std::accumulate(xs.begin(), xs.end(), 0.0) / static_cast<double>(xs.size());
}

double sample_stddev(const std::vector<double>& xs, double m) {
    if (xs.size() < 2) {
        return 0.0;
    }
    double acc = 0.0;
    for (double x : xs) {
        double d = x - m;
        acc += d * d;
    }
    return std::sqrt(acc / static_cast<double>(xs.size() - 1));
}

double recall_at_k(const std::vector<int>& truth, const std::vector<int>& pred, int k) {
    std::unordered_set<int> truth_set;
    for (int i = 0; i < k && i < static_cast<int>(truth.size()); ++i) {
        truth_set.insert(truth[static_cast<size_t>(i)]);
    }
    int hit = 0;
    for (int i = 0; i < k && i < static_cast<int>(pred.size()); ++i) {
        if (truth_set.count(pred[static_cast<size_t>(i)]) > 0U) {
            ++hit;
        }
    }
    return static_cast<double>(hit) / static_cast<double>(k);
}

int main() {
    const int d = 10000;
    const int n = 30000;
    const int k = 10;
    const int samples = 8;
    const int queries_per_sample = 8;
    const int warmup_queries = 1;
    const int posting_cap = 200;
    const int candidate_cap = 2000;
    const double ci95_z = 1.96;

    const std::vector<int> s_values = {10, 20, 40, 80, 160, 320};

    std::mt19937_64 rng(20260402ULL);
    std::uniform_int_distribution<int> q_dist(0, n - 1);

    std::ofstream csv("sparse_advanced_results.csv");
    if (!csv.is_open()) {
        std::cerr << "cannot open sparse_advanced_results.csv" << std::endl;
        return 1;
    }

    csv << "s,method,avg_query_ms,stddev_ms,ci95_ms,avg_candidates,avg_recall_at_k,avg_index_build_ms\n";

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "n=" << n << ", d=" << d << ", k=" << k << ", samples=" << samples
              << ", queries_per_sample=" << queries_per_sample << std::endl;

    for (int s : s_values) {
        std::vector<double> t_baseline;
        std::vector<double> t_inverted;
        std::vector<double> t_two_stage;
        std::vector<double> c_baseline;
        std::vector<double> c_inverted;
        std::vector<double> c_two_stage;
        std::vector<double> recall_two_stage;
        std::vector<double> index_build_times;

        for (int rep = 0; rep < samples; ++rep) {
            auto users = generate_sparse_users(n, d, s, rng);

            auto idx_t0 = std::chrono::high_resolution_clock::now();
            auto inv = build_inverted_index(users, d);
            auto idx_t1 = std::chrono::high_resolution_clock::now();
            index_build_times.push_back(
                std::chrono::duration<double, std::milli>(idx_t1 - idx_t0).count());

            for (int w = 0; w < warmup_queries; ++w) {
                int q = q_dist(rng);
                (void)query_full_scan_heap(users, q, k);
                (void)query_inverted_exact(users, inv, q, k);
                (void)query_two_stage(users, inv, q, k, posting_cap, candidate_cap);
            }

            for (int qi = 0; qi < queries_per_sample; ++qi) {
                int q = q_dist(rng);

                QueryOutput out_baseline = query_full_scan_heap(users, q, k);
                QueryOutput out_inverted = query_inverted_exact(users, inv, q, k);
                QueryOutput out_two_stage =
                    query_two_stage(users, inv, q, k, posting_cap, candidate_cap);

                t_baseline.push_back(out_baseline.ms);
                t_inverted.push_back(out_inverted.ms);
                t_two_stage.push_back(out_two_stage.ms);

                c_baseline.push_back(static_cast<double>(out_baseline.candidates));
                c_inverted.push_back(static_cast<double>(out_inverted.candidates));
                c_two_stage.push_back(static_cast<double>(out_two_stage.candidates));

                recall_two_stage.push_back(recall_at_k(out_inverted.topk_users, out_two_stage.topk_users, k));
            }
        }

        auto write_row = [&](const std::string& method,
                             const std::vector<double>& times,
                             const std::vector<double>& candidates,
                             const std::vector<double>& recalls,
                             double idx_ms) {
            double m = mean(times);
            double sd = sample_stddev(times, m);
            double ci95 = ci95_z * (sd / std::sqrt(static_cast<double>(times.size())));
            double cand = mean(candidates);
            double rec = recalls.empty() ? 1.0 : mean(recalls);
            csv << s << ',' << method << ',' << m << ',' << sd << ',' << ci95 << ',' << cand << ',' << rec << ','
                << idx_ms << '\n';
        };

        const double avg_idx_ms = mean(index_build_times);
        write_row("baseline_full_scan_heap", t_baseline, c_baseline, {}, avg_idx_ms);
        write_row("inverted_exact", t_inverted, c_inverted, {}, avg_idx_ms);
        write_row("two_stage_recall_rerank", t_two_stage, c_two_stage, recall_two_stage, avg_idx_ms);

        std::cout << "s=" << std::setw(4) << s << "  baseline=" << mean(t_baseline) << " ms"
                  << "  inverted=" << mean(t_inverted) << " ms"
                  << "  two_stage=" << mean(t_two_stage) << " ms"
                  << "  recall@" << k << "=" << mean(recall_two_stage)
                  << "  idx_build=" << avg_idx_ms << " ms" << std::endl;
    }

    std::cout << "done -> sparse_advanced_results.csv" << std::endl;
    return 0;
}
