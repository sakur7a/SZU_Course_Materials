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

struct EntryD {
    int idx;
    double val;
};

struct UserAOS {
    std::vector<EntryD> entries;
    double norm = 0.0;
    double inv_norm = 0.0;
};

struct PostingD {
    int user;
    double val;
};

struct CSRFloat {
    std::vector<int> indptr;
    std::vector<int> indices;
    std::vector<float> values;
    std::vector<float> norm;
    std::vector<float> inv_norm;
    int n = 0;
};

struct PostingF {
    int user;
    float val;
};

struct QueryResult {
    double ms = 0.0;
    std::vector<int> topk;
    int candidates = 0;
};

static std::vector<UserAOS> generate_users_aos(int n, int d, int s, std::mt19937_64& rng) {
    std::uniform_int_distribution<int> idx_dist(0, d - 1);
    std::uniform_real_distribution<double> val_dist(0.0, 1.0);

    std::vector<UserAOS> users(static_cast<size_t>(n));
    for (int u = 0; u < n; ++u) {
        std::unordered_set<int> picked;
        picked.reserve(static_cast<size_t>(s) * 2U);
        while (static_cast<int>(picked.size()) < s) {
            picked.insert(idx_dist(rng));
        }

        std::vector<int> idxs(picked.begin(), picked.end());
        std::sort(idxs.begin(), idxs.end());

        users[static_cast<size_t>(u)].entries.reserve(static_cast<size_t>(s));
        double sq = 0.0;
        for (int idx : idxs) {
            double v = val_dist(rng);
            users[static_cast<size_t>(u)].entries.push_back({idx, v});
            sq += v * v;
        }
        users[static_cast<size_t>(u)].norm = std::sqrt(sq);
        users[static_cast<size_t>(u)].inv_norm =
            users[static_cast<size_t>(u)].norm > 0.0 ? (1.0 / users[static_cast<size_t>(u)].norm) : 0.0;
    }
    return users;
}

static double dot_aos(const UserAOS& a, const UserAOS& b) {
    size_t i = 0;
    size_t j = 0;
    double dot = 0.0;
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

static std::vector<std::vector<PostingD>> build_inverted_aos(const std::vector<UserAOS>& users, int d) {
    std::vector<std::vector<PostingD>> inv(static_cast<size_t>(d));
    for (int u = 0; u < static_cast<int>(users.size()); ++u) {
        for (const auto& e : users[static_cast<size_t>(u)].entries) {
            inv[static_cast<size_t>(e.idx)].push_back({u, e.val});
        }
    }
    return inv;
}

static CSRFloat build_csr_float(const std::vector<UserAOS>& users) {
    CSRFloat csr;
    csr.n = static_cast<int>(users.size());
    csr.indptr.resize(static_cast<size_t>(csr.n + 1), 0);
    csr.norm.resize(static_cast<size_t>(csr.n), 0.0F);
    csr.inv_norm.resize(static_cast<size_t>(csr.n), 0.0F);

    int total_nnz = 0;
    for (int u = 0; u < csr.n; ++u) {
        total_nnz += static_cast<int>(users[static_cast<size_t>(u)].entries.size());
        csr.indptr[static_cast<size_t>(u + 1)] = total_nnz;
    }

    csr.indices.resize(static_cast<size_t>(total_nnz));
    csr.values.resize(static_cast<size_t>(total_nnz));

    int ptr = 0;
    for (int u = 0; u < csr.n; ++u) {
        for (const auto& e : users[static_cast<size_t>(u)].entries) {
            csr.indices[static_cast<size_t>(ptr)] = e.idx;
            csr.values[static_cast<size_t>(ptr)] = static_cast<float>(e.val);
            ++ptr;
        }
        float nrm = static_cast<float>(users[static_cast<size_t>(u)].norm);
        csr.norm[static_cast<size_t>(u)] = nrm;
        csr.inv_norm[static_cast<size_t>(u)] = (nrm > 0.0F) ? (1.0F / nrm) : 0.0F;
    }

    return csr;
}

static std::vector<std::vector<PostingF>> build_inverted_csr(const CSRFloat& csr, int d) {
    std::vector<std::vector<PostingF>> inv(static_cast<size_t>(d));
    for (int u = 0; u < csr.n; ++u) {
        int st = csr.indptr[static_cast<size_t>(u)];
        int ed = csr.indptr[static_cast<size_t>(u + 1)];
        for (int p = st; p < ed; ++p) {
            inv[static_cast<size_t>(csr.indices[static_cast<size_t>(p)])].push_back(
                {u, csr.values[static_cast<size_t>(p)]});
        }
    }
    return inv;
}

static float dot_csr(const CSRFloat& csr, int u, int v) {
    int i = csr.indptr[static_cast<size_t>(u)];
    int i_end = csr.indptr[static_cast<size_t>(u + 1)];
    int j = csr.indptr[static_cast<size_t>(v)];
    int j_end = csr.indptr[static_cast<size_t>(v + 1)];
    float dot = 0.0F;

    while (i < i_end && j < j_end) {
        int idx_i = csr.indices[static_cast<size_t>(i)];
        int idx_j = csr.indices[static_cast<size_t>(j)];
        if (idx_i == idx_j) {
            dot += csr.values[static_cast<size_t>(i)] * csr.values[static_cast<size_t>(j)];
            ++i;
            ++j;
        } else if (idx_i < idx_j) {
            ++i;
        } else {
            ++j;
        }
    }
    return dot;
}

static QueryResult query_baseline_scan_aos(const std::vector<UserAOS>& users, int target, int k) {
    auto t0 = std::chrono::high_resolution_clock::now();

    auto cmp = [](const std::pair<double, int>& a, const std::pair<double, int>& b) { return a.first > b.first; };
    std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>, decltype(cmp)> topk(cmp);

    const auto& tu = users[static_cast<size_t>(target)];
    for (int u = 0; u < static_cast<int>(users.size()); ++u) {
        if (u == target) {
            continue;
        }
        double dot = dot_aos(tu, users[static_cast<size_t>(u)]);
        double sim = dot / (tu.norm * users[static_cast<size_t>(u)].norm);

        if (static_cast<int>(topk.size()) < k) {
            topk.push({sim, u});
        } else if (sim > topk.top().first) {
            topk.pop();
            topk.push({sim, u});
        }
    }

    std::vector<int> out;
    out.reserve(static_cast<size_t>(k));
    while (!topk.empty()) {
        out.push_back(topk.top().second);
        topk.pop();
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return {ms, out, static_cast<int>(users.size()) - 1};
}

static QueryResult query_code_only_scan_csr(const CSRFloat& csr, int target, int k) {
    auto t0 = std::chrono::high_resolution_clock::now();

    auto cmp = [](const std::pair<float, int>& a, const std::pair<float, int>& b) { return a.first > b.first; };
    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, decltype(cmp)> topk(cmp);

    float inv_t = csr.inv_norm[static_cast<size_t>(target)];
    for (int u = 0; u < csr.n; ++u) {
        if (u == target) {
            continue;
        }
        float dot = dot_csr(csr, target, u);
        float sim = dot * inv_t * csr.inv_norm[static_cast<size_t>(u)];

        if (static_cast<int>(topk.size()) < k) {
            topk.push({sim, u});
        } else if (sim > topk.top().first) {
            topk.pop();
            topk.push({sim, u});
        }
    }

    std::vector<int> out;
    out.reserve(static_cast<size_t>(k));
    while (!topk.empty()) {
        out.push_back(topk.top().second);
        topk.pop();
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return {ms, out, csr.n - 1};
}

static QueryResult query_algo_only_inverted_aos(const std::vector<UserAOS>& users,
                                                 const std::vector<std::vector<PostingD>>& inv,
                                                 int target,
                                                 int k) {
    auto t0 = std::chrono::high_resolution_clock::now();

    const auto& tu = users[static_cast<size_t>(target)];
    std::unordered_map<int, double> dot_map;
    dot_map.reserve(4096);

    for (const auto& e : tu.entries) {
        const auto& postings = inv[static_cast<size_t>(e.idx)];
        for (const auto& p : postings) {
            if (p.user == target) {
                continue;
            }
            dot_map[p.user] += e.val * p.val;
        }
    }

    auto cmp = [](const std::pair<double, int>& a, const std::pair<double, int>& b) { return a.first > b.first; };
    std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>, decltype(cmp)> topk(cmp);

    for (const auto& kv : dot_map) {
        int u = kv.first;
        double sim = kv.second / (tu.norm * users[static_cast<size_t>(u)].norm);
        if (static_cast<int>(topk.size()) < k) {
            topk.push({sim, u});
        } else if (sim > topk.top().first) {
            topk.pop();
            topk.push({sim, u});
        }
    }

    std::vector<int> out;
    out.reserve(static_cast<size_t>(k));
    while (!topk.empty()) {
        out.push_back(topk.top().second);
        topk.pop();
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return {ms, out, static_cast<int>(dot_map.size())};
}

static QueryResult query_combined_inverted_csr(const CSRFloat& csr,
                                                const std::vector<std::vector<PostingF>>& inv,
                                                int target,
                                                int k) {
    auto t0 = std::chrono::high_resolution_clock::now();

    int st = csr.indptr[static_cast<size_t>(target)];
    int ed = csr.indptr[static_cast<size_t>(target + 1)];

    std::unordered_map<int, float> dot_map;
    dot_map.reserve(4096);

    for (int p = st; p < ed; ++p) {
        int idx = csr.indices[static_cast<size_t>(p)];
        float v = csr.values[static_cast<size_t>(p)];
        const auto& postings = inv[static_cast<size_t>(idx)];
        for (const auto& it : postings) {
            if (it.user == target) {
                continue;
            }
            dot_map[it.user] += v * it.val;
        }
    }

    auto cmp = [](const std::pair<float, int>& a, const std::pair<float, int>& b) { return a.first > b.first; };
    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, decltype(cmp)> topk(cmp);

    float inv_t = csr.inv_norm[static_cast<size_t>(target)];
    for (const auto& kv : dot_map) {
        int u = kv.first;
        float sim = kv.second * inv_t * csr.inv_norm[static_cast<size_t>(u)];
        if (static_cast<int>(topk.size()) < k) {
            topk.push({sim, u});
        } else if (sim > topk.top().first) {
            topk.pop();
            topk.push({sim, u});
        }
    }

    std::vector<int> out;
    out.reserve(static_cast<size_t>(k));
    while (!topk.empty()) {
        out.push_back(topk.top().second);
        topk.pop();
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return {ms, out, static_cast<int>(dot_map.size())};
}

static double mean(const std::vector<double>& xs) {
    if (xs.empty()) {
        return 0.0;
    }
    return std::accumulate(xs.begin(), xs.end(), 0.0) / static_cast<double>(xs.size());
}

static double sample_stddev(const std::vector<double>& xs, double m) {
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

static double overlap_at_k(const std::vector<int>& ref, const std::vector<int>& cur, int k) {
    std::unordered_set<int> s;
    for (int i = 0; i < k && i < static_cast<int>(ref.size()); ++i) {
        s.insert(ref[static_cast<size_t>(i)]);
    }
    int hit = 0;
    for (int i = 0; i < k && i < static_cast<int>(cur.size()); ++i) {
        if (s.count(cur[static_cast<size_t>(i)]) > 0U) {
            ++hit;
        }
    }
    return static_cast<double>(hit) / static_cast<double>(k);
}

int main() {
    const int n = 20000;
    const int d = 10000;
    const int k = 10;
    const int samples = 6;
    const int queries_per_sample = 6;
    const int warmup = 1;
    const double z95 = 1.96;

    const std::vector<int> s_values = {20, 40, 80, 160, 320};

    std::mt19937_64 rng(20260402ULL);
    std::uniform_int_distribution<int> qdist(0, n - 1);

    std::ofstream csv("results/advanced/unified_opt_results.csv");
    if (!csv.is_open()) {
        std::cerr << "cannot create results/advanced/unified_opt_results.csv" << std::endl;
        return 1;
    }
    csv << "s,method,avg_ms,stddev_ms,ci95_ms,speedup_vs_baseline,avg_candidates,avg_overlap_at_k\n";

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "n=" << n << ", d=" << d << ", k=" << k << std::endl;

    for (int s : s_values) {
        std::vector<double> t_base;
        std::vector<double> t_code;
        std::vector<double> t_algo;
        std::vector<double> t_comb;
        std::vector<double> c_base;
        std::vector<double> c_code;
        std::vector<double> c_algo;
        std::vector<double> c_comb;
        std::vector<double> ov_code;
        std::vector<double> ov_algo;
        std::vector<double> ov_comb;

        for (int rep = 0; rep < samples; ++rep) {
            auto users = generate_users_aos(n, d, s, rng);
            auto inv_aos = build_inverted_aos(users, d);
            auto csr = build_csr_float(users);
            auto inv_csr = build_inverted_csr(csr, d);

            for (int w = 0; w < warmup; ++w) {
                int q = qdist(rng);
                (void)query_baseline_scan_aos(users, q, k);
                (void)query_code_only_scan_csr(csr, q, k);
                (void)query_algo_only_inverted_aos(users, inv_aos, q, k);
                (void)query_combined_inverted_csr(csr, inv_csr, q, k);
            }

            for (int qi = 0; qi < queries_per_sample; ++qi) {
                int q = qdist(rng);
                QueryResult r_base = query_baseline_scan_aos(users, q, k);
                QueryResult r_code = query_code_only_scan_csr(csr, q, k);
                QueryResult r_algo = query_algo_only_inverted_aos(users, inv_aos, q, k);
                QueryResult r_comb = query_combined_inverted_csr(csr, inv_csr, q, k);

                t_base.push_back(r_base.ms);
                t_code.push_back(r_code.ms);
                t_algo.push_back(r_algo.ms);
                t_comb.push_back(r_comb.ms);

                c_base.push_back(static_cast<double>(r_base.candidates));
                c_code.push_back(static_cast<double>(r_code.candidates));
                c_algo.push_back(static_cast<double>(r_algo.candidates));
                c_comb.push_back(static_cast<double>(r_comb.candidates));

                ov_code.push_back(overlap_at_k(r_base.topk, r_code.topk, k));
                ov_algo.push_back(overlap_at_k(r_base.topk, r_algo.topk, k));
                ov_comb.push_back(overlap_at_k(r_base.topk, r_comb.topk, k));
            }
        }

        double mb = mean(t_base);
        auto write_row = [&](const std::string& method,
                             const std::vector<double>& t,
                             const std::vector<double>& c,
                             const std::vector<double>& ov) {
            double m = mean(t);
            double sd = sample_stddev(t, m);
            double ci = z95 * (sd / std::sqrt(static_cast<double>(t.size())));
            double speedup = mb / m;
            double cand = mean(c);
            double overlap = ov.empty() ? 1.0 : mean(ov);
            csv << s << ',' << method << ',' << m << ',' << sd << ',' << ci << ',' << speedup << ',' << cand << ','
                << overlap << '\n';
        };

        write_row("baseline_scan_aos_double", t_base, c_base, {});
        write_row("code_only_scan_csr_float", t_code, c_code, ov_code);
        write_row("algo_only_inverted_aos", t_algo, c_algo, ov_algo);
        write_row("combined_inverted_csr_float", t_comb, c_comb, ov_comb);

        std::cout << "s=" << std::setw(4) << s << "  baseline=" << mean(t_base)
                  << " ms  code_only=" << mean(t_code) << " ms"
                  << "  algo_only=" << mean(t_algo) << " ms"
                  << "  combined=" << mean(t_comb) << " ms" << std::endl;
    }

    std::cout << "done -> results/advanced/unified_opt_results.csv" << std::endl;
    return 0;
}
