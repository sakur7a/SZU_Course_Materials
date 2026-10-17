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
};

struct CSRFloat {
    std::vector<int> indptr;
    std::vector<int> indices;
    std::vector<float> values;
    std::vector<float> norm;
    std::vector<float> inv_norm;
    int n = 0;
};

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
    }
    return users;
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

static std::vector<int> topk_from_pairs(std::vector<std::pair<double, int>>& pairs, int k) {
    std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    std::vector<int> out;
    int m = std::min(k, static_cast<int>(pairs.size()));
    out.reserve(static_cast<size_t>(m));
    for (int i = 0; i < m; ++i) {
        out.push_back(pairs[static_cast<size_t>(i)].second);
    }
    return out;
}

static std::vector<int> query_baseline_aos(const std::vector<UserAOS>& users, int target, int k, double& elapsed_ms) {
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<std::pair<double, int>> sims;
    sims.reserve(users.size() - 1);
    const auto& tu = users[static_cast<size_t>(target)];
    for (int u = 0; u < static_cast<int>(users.size()); ++u) {
        if (u == target) {
            continue;
        }
        double dot = dot_aos(tu, users[static_cast<size_t>(u)]);
        double sim = dot / (tu.norm * users[static_cast<size_t>(u)].norm);  // baseline: per-query divisions
        sims.push_back({sim, u});
    }

    auto out = topk_from_pairs(sims, k);

    auto t1 = std::chrono::high_resolution_clock::now();
    elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return out;
}

struct CSRQueryRunner {
    std::vector<float> sim_buffer;             // reused buffer for all users
    std::vector<std::pair<float, int>> pairs;  // reused top-k staging

    explicit CSRQueryRunner(int n) {
        sim_buffer.resize(static_cast<size_t>(n), -1.0F);
        pairs.reserve(static_cast<size_t>(n));
    }

    std::vector<int> query(const CSRFloat& csr, int target, int k, double& elapsed_ms) {
        auto t0 = std::chrono::high_resolution_clock::now();

        pairs.clear();
        pairs.reserve(static_cast<size_t>(csr.n - 1));

        const float inv_t = csr.inv_norm[static_cast<size_t>(target)];
        for (int u = 0; u < csr.n; ++u) {
            if (u == target) {
                continue;
            }
            float dot = dot_csr(csr, target, u);
            // optimized: precomputed inverse norms + float pipeline
            float sim = dot * inv_t * csr.inv_norm[static_cast<size_t>(u)];
            sim_buffer[static_cast<size_t>(u)] = sim;
            pairs.push_back({sim, u});
        }

        std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        std::vector<int> out;
        int m = std::min(k, static_cast<int>(pairs.size()));
        out.reserve(static_cast<size_t>(m));
        for (int i = 0; i < m; ++i) {
            out.push_back(pairs[static_cast<size_t>(i)].second);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return out;
    }
};

static double overlap_at_k(const std::vector<int>& a, const std::vector<int>& b, int k) {
    std::unordered_set<int> sa;
    for (int i = 0; i < k && i < static_cast<int>(a.size()); ++i) {
        sa.insert(a[static_cast<size_t>(i)]);
    }
    int hit = 0;
    for (int i = 0; i < k && i < static_cast<int>(b.size()); ++i) {
        if (sa.count(b[static_cast<size_t>(i)]) > 0U) {
            ++hit;
        }
    }
    return static_cast<double>(hit) / static_cast<double>(k);
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

int main() {
    const int d = 10000;
    const int n = 20000;
    const int k = 10;
    const int samples = 8;
    const int queries_per_sample = 8;
    const int warmup = 1;
    const double z95 = 1.96;

    const std::vector<int> s_values = {20, 40, 80, 160, 320};

    std::mt19937_64 rng(20260402ULL);
    std::uniform_int_distribution<int> qdist(0, n - 1);

    std::ofstream csv("results/advanced/sparse_lowlevel_opt_results.csv");
    if (!csv.is_open()) {
        std::cerr << "cannot create results/advanced/sparse_lowlevel_opt_results.csv" << std::endl;
        return 1;
    }
    csv << "s,method,avg_ms,stddev_ms,ci95_ms,overlap_at_k\n";

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "n=" << n << ", d=" << d << ", k=" << k << std::endl;

    for (int s : s_values) {
        std::vector<double> t_base;
        std::vector<double> t_opt;
        std::vector<double> overlaps;

        for (int rep = 0; rep < samples; ++rep) {
            auto users_aos = generate_users_aos(n, d, s, rng);
            auto csr = build_csr_float(users_aos);
            CSRQueryRunner runner(n);

            for (int w = 0; w < warmup; ++w) {
                int q = qdist(rng);
                double tb = 0.0;
                double to = 0.0;
                (void)query_baseline_aos(users_aos, q, k, tb);
                (void)runner.query(csr, q, k, to);
            }

            for (int qi = 0; qi < queries_per_sample; ++qi) {
                int q = qdist(rng);
                double tb = 0.0;
                double to = 0.0;
                auto top_base = query_baseline_aos(users_aos, q, k, tb);
                auto top_opt = runner.query(csr, q, k, to);

                t_base.push_back(tb);
                t_opt.push_back(to);
                overlaps.push_back(overlap_at_k(top_base, top_opt, k));
            }
        }

        double mb = mean(t_base);
        double sdb = sample_stddev(t_base, mb);
        double cib = z95 * (sdb / std::sqrt(static_cast<double>(t_base.size())));

        double mo = mean(t_opt);
        double sdo = sample_stddev(t_opt, mo);
        double cio = z95 * (sdo / std::sqrt(static_cast<double>(t_opt.size())));

        double ov = mean(overlaps);

        csv << s << ",baseline_aos_double," << mb << ',' << sdb << ',' << cib << ",1.000\n";
        csv << s << ",optimized_csr_float_invnorm," << mo << ',' << sdo << ',' << cio << ',' << ov << "\n";

        std::cout << "s=" << std::setw(4) << s << "  baseline=" << mb << " ms"
                  << "  optimized=" << mo << " ms"
                  << "  speedup=" << (mb / mo) << "x"
                  << "  overlap@" << k << "=" << ov << std::endl;
    }

    std::cout << "done -> results/advanced/sparse_lowlevel_opt_results.csv" << std::endl;
    return 0;
}
