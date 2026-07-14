#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

struct Point {
    double x;
    double y;
    int id;
};

struct ClosestPairResult {
    int a;
    int b;
    double dist2;
};

struct TraceEvent {
    int step;
    int id_a;
    int id_b;
    double dist2;
    std::string phase;
};

struct DivisionStep {
    int depth;
    int left;
    int right;
    double mid_x;
    double strip_left;
    double strip_right;
    double best_dist;
};

static double distance2(const Point &p1, const Point &p2) {
    const double dx = p1.x - p2.x;
    const double dy = p1.y - p2.y;
    return dx * dx + dy * dy;
}

static ClosestPairResult better(const ClosestPairResult &lhs, const ClosestPairResult &rhs) {
    if (lhs.dist2 < rhs.dist2) {
        return lhs;
    }
    return rhs;
}

static std::vector<Point> generate_points(std::size_t n, std::uint64_t seed, double min_coord, double max_coord) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(min_coord, max_coord);

    std::vector<Point> points;
    points.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        Point p;
        p.x = dist(rng);
        p.y = dist(rng);
        p.id = static_cast<int>(i);
        points.push_back(p);
    }
    return points;
}

static ClosestPairResult brute_force_range(const std::vector<Point> &points, int left, int right) {
    ClosestPairResult best;
    best.a = -1;
    best.b = -1;
    best.dist2 = std::numeric_limits<double>::infinity();

    for (int i = left; i < right; ++i) {
        for (int j = i + 1; j < right; ++j) {
            const double d2 = distance2(points[i], points[j]);
            if (d2 < best.dist2) {
                best.a = points[i].id;
                best.b = points[j].id;
                best.dist2 = d2;
            }
        }
    }
    return best;
}

static ClosestPairResult brute_force(const std::vector<Point> &points) {
    return brute_force_range(points, 0, static_cast<int>(points.size()));
}

static ClosestPairResult divide_conquer_recursive(
    const std::vector<Point> &points_by_x,
    const std::vector<int> &indices_sorted_by_y,
    int left,
    int right
) {
    const int count = right - left;
    if (count <= 3) {
        std::vector<Point> small;
        small.reserve(static_cast<std::size_t>(count));
        for (int i = left; i < right; ++i) {
            small.push_back(points_by_x[i]);
        }
        return brute_force(small);
    }

    const int mid = left + count / 2;
    const double mid_x = points_by_x[mid].x;

    std::vector<int> left_y;
    std::vector<int> right_y;
    left_y.reserve(indices_sorted_by_y.size());
    right_y.reserve(indices_sorted_by_y.size());

    for (std::size_t i = 0; i < indices_sorted_by_y.size(); ++i) {
        const int idx = indices_sorted_by_y[i];
        if (idx >= left && idx < mid) {
            left_y.push_back(idx);
        } else if (idx >= mid && idx < right) {
            right_y.push_back(idx);
        }
    }

    const ClosestPairResult left_best = divide_conquer_recursive(points_by_x, left_y, left, mid);
    const ClosestPairResult right_best = divide_conquer_recursive(points_by_x, right_y, mid, right);
    ClosestPairResult best = better(left_best, right_best);

    std::vector<int> strip;
    strip.reserve(indices_sorted_by_y.size());
    for (std::size_t i = 0; i < indices_sorted_by_y.size(); ++i) {
        const int idx = indices_sorted_by_y[i];
        const double dx = points_by_x[idx].x - mid_x;
        if (dx * dx < best.dist2) {
            strip.push_back(idx);
        }
    }

    // In y-sorted strip, each point only needs to compare to a constant number of following points.
    for (std::size_t i = 0; i < strip.size(); ++i) {
        for (std::size_t j = i + 1; j < strip.size(); ++j) {
            const double dy = points_by_x[strip[j]].y - points_by_x[strip[i]].y;
            if (dy * dy >= best.dist2) {
                break;
            }
            const double d2 = distance2(points_by_x[strip[i]], points_by_x[strip[j]]);
            if (d2 < best.dist2) {
                best.a = points_by_x[strip[i]].id;
                best.b = points_by_x[strip[j]].id;
                best.dist2 = d2;
            }
        }
    }

    return best;
}

static ClosestPairResult divide_conquer_recursive_trace(
    const std::vector<Point> &points_by_x,
    const std::vector<int> &indices_sorted_by_y,
    int left,
    int right,
    std::vector<TraceEvent> &events,
    int &step_counter
) {
    const int count = right - left;
    if (count <= 3) {
        ClosestPairResult best;
        best.a = -1;
        best.b = -1;
        best.dist2 = std::numeric_limits<double>::infinity();

        for (int i = left; i < right; ++i) {
            for (int j = i + 1; j < right; ++j) {
                const double d2 = distance2(points_by_x[i], points_by_x[j]);
                if (d2 < best.dist2) {
                    best.a = points_by_x[i].id;
                    best.b = points_by_x[j].id;
                    best.dist2 = d2;

                    TraceEvent e;
                    e.step = ++step_counter;
                    e.id_a = best.a;
                    e.id_b = best.b;
                    e.dist2 = best.dist2;
                    e.phase = "base";
                    events.push_back(e);
                }
            }
        }
        return best;
    }

    const int mid = left + count / 2;
    const double mid_x = points_by_x[mid].x;

    std::vector<int> left_y;
    std::vector<int> right_y;
    left_y.reserve(indices_sorted_by_y.size());
    right_y.reserve(indices_sorted_by_y.size());

    for (std::size_t i = 0; i < indices_sorted_by_y.size(); ++i) {
        const int idx = indices_sorted_by_y[i];
        if (idx >= left && idx < mid) {
            left_y.push_back(idx);
        } else if (idx >= mid && idx < right) {
            right_y.push_back(idx);
        }
    }

    const ClosestPairResult left_best = divide_conquer_recursive_trace(points_by_x, left_y, left, mid, events, step_counter);
    const ClosestPairResult right_best = divide_conquer_recursive_trace(points_by_x, right_y, mid, right, events, step_counter);
    ClosestPairResult best = better(left_best, right_best);

    std::vector<int> strip;
    strip.reserve(indices_sorted_by_y.size());
    for (std::size_t i = 0; i < indices_sorted_by_y.size(); ++i) {
        const int idx = indices_sorted_by_y[i];
        const double dx = points_by_x[idx].x - mid_x;
        if (dx * dx < best.dist2) {
            strip.push_back(idx);
        }
    }

    for (std::size_t i = 0; i < strip.size(); ++i) {
        for (std::size_t j = i + 1; j < strip.size(); ++j) {
            const double dy = points_by_x[strip[j]].y - points_by_x[strip[i]].y;
            if (dy * dy >= best.dist2) {
                break;
            }
            const double d2 = distance2(points_by_x[strip[i]], points_by_x[strip[j]]);
            if (d2 < best.dist2) {
                best.a = points_by_x[strip[i]].id;
                best.b = points_by_x[strip[j]].id;
                best.dist2 = d2;

                TraceEvent e;
                e.step = ++step_counter;
                e.id_a = best.a;
                e.id_b = best.b;
                e.dist2 = best.dist2;
                e.phase = "strip";
                events.push_back(e);
            }
        }
    }

    return best;
}

static ClosestPairResult divide_conquer(std::vector<Point> points) {
    if (points.size() < 2) {
        ClosestPairResult r;
        r.a = -1;
        r.b = -1;
        r.dist2 = std::numeric_limits<double>::infinity();
        return r;
    }

    std::sort(points.begin(), points.end(), [](const Point &lhs, const Point &rhs) {
        if (lhs.x != rhs.x) {
            return lhs.x < rhs.x;
        }
        if (lhs.y != rhs.y) {
            return lhs.y < rhs.y;
        }
        return lhs.id < rhs.id;
    });

    std::vector<int> by_y(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        by_y[i] = static_cast<int>(i);
    }
    std::sort(by_y.begin(), by_y.end(), [&](int a, int b) {
        if (points[a].y != points[b].y) {
            return points[a].y < points[b].y;
        }
        if (points[a].x != points[b].x) {
            return points[a].x < points[b].x;
        }
        return points[a].id < points[b].id;
    });

    return divide_conquer_recursive(points, by_y, 0, static_cast<int>(points.size()));
}

static ClosestPairResult divide_conquer_with_trace(std::vector<Point> points, std::vector<TraceEvent> *events_out) {
    if (points.size() < 2) {
        ClosestPairResult r;
        r.a = -1;
        r.b = -1;
        r.dist2 = std::numeric_limits<double>::infinity();
        return r;
    }

    std::sort(points.begin(), points.end(), [](const Point &lhs, const Point &rhs) {
        if (lhs.x != rhs.x) {
            return lhs.x < rhs.x;
        }
        if (lhs.y != rhs.y) {
            return lhs.y < rhs.y;
        }
        return lhs.id < rhs.id;
    });

    std::vector<int> by_y(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        by_y[i] = static_cast<int>(i);
    }
    std::sort(by_y.begin(), by_y.end(), [&](int a, int b) {
        if (points[a].y != points[b].y) {
            return points[a].y < points[b].y;
        }
        if (points[a].x != points[b].x) {
            return points[a].x < points[b].x;
        }
        return points[a].id < points[b].id;
    });

    std::vector<TraceEvent> local_events;
    int step_counter = 0;
    ClosestPairResult res = divide_conquer_recursive_trace(points, by_y, 0, static_cast<int>(points.size()), local_events, step_counter);

    if (events_out != nullptr) {
        *events_out = local_events;
    }
    return res;
}

static void save_points_csv(const std::string &path, const std::vector<Point> &points) {
    std::ofstream out(path.c_str());
    out << "id,x,y\n";
    out << std::setprecision(17);
    for (std::size_t i = 0; i < points.size(); ++i) {
        out << points[i].id << ',' << points[i].x << ',' << points[i].y << '\n';
    }
}

static void save_pair_csv(const std::string &path, const ClosestPairResult &res, const std::vector<Point> &points) {
    if (res.a < 0 || res.b < 0) {
        return;
    }

    const Point &pa = points[static_cast<std::size_t>(res.a)];
    const Point &pb = points[static_cast<std::size_t>(res.b)];

    std::ofstream out(path.c_str());
    out << "id_a,id_b,ax,ay,bx,by,distance\n";
    out << std::setprecision(17);
    out << res.a << ',' << res.b << ','
        << pa.x << ',' << pa.y << ','
        << pb.x << ',' << pb.y << ','
        << std::sqrt(res.dist2) << '\n';
}

static void save_trace_csv(const std::string &path, const std::vector<TraceEvent> &events) {
    std::ofstream out(path.c_str());
    out << "step,id_a,id_b,distance,phase\n";
    out << std::setprecision(17);
    for (std::size_t i = 0; i < events.size(); ++i) {
        out << events[i].step << ','
            << events[i].id_a << ','
            << events[i].id_b << ','
            << std::sqrt(events[i].dist2) << ','
            << events[i].phase
            << '\n';
    }
}

static void save_division_csv(const std::string &path, const std::vector<DivisionStep> &steps) {
    std::ofstream out(path.c_str());
    out << "depth,left,right,mid_x,strip_left,strip_right,best_dist\n";
    out << std::setprecision(17);
    for (std::size_t i = 0; i < steps.size(); ++i) {
        out << steps[i].depth << ','
            << steps[i].left << ','
            << steps[i].right << ','
            << steps[i].mid_x << ','
            << steps[i].strip_left << ','
            << steps[i].strip_right << ','
            << steps[i].best_dist
            << '\n';
    }
}

struct BenchRow {
    std::size_t n;
    std::string algorithm;
    int run_idx;
    double ms;
    bool extrapolated;
};

static volatile double g_result_sink = 0.0;

static std::string get_arg_value(const std::vector<std::string> &args, const std::string &key, const std::string &default_val) {
    for (std::size_t i = 0; i + 1 < args.size(); ++i) {
        if (args[i] == key) {
            return args[i + 1];
        }
    }
    return default_val;
}

static bool has_flag(const std::vector<std::string> &args, const std::string &flag) {
    for (std::size_t i = 0; i < args.size(); ++i) {
        if (args[i] == flag) {
            return true;
        }
    }
    return false;
}

static double now_ms() {
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count()
    ) / 1000.0;
}

static double measure_bruteforce_ms(const std::vector<Point> &points, ClosestPairResult *out_res) {
    const double start = now_ms();
    const ClosestPairResult res = brute_force(points);
    const double end = now_ms();
    g_result_sink += res.dist2;
    if (out_res != nullptr) {
        *out_res = res;
    }
    return end - start;
}

static double measure_divide_ms(const std::vector<Point> &points, ClosestPairResult *out_res) {
    const double start = now_ms();
    const ClosestPairResult res = divide_conquer(points);
    const double end = now_ms();
    g_result_sink += res.dist2;
    if (out_res != nullptr) {
        *out_res = res;
    }
    return end - start;
}

static void write_benchmark_csv(const std::string &path, const std::vector<BenchRow> &rows) {
    std::ofstream out(path.c_str());
    out << "n,algorithm,run_idx,time_ms,extrapolated\n";
    out << std::fixed << std::setprecision(6);
    for (std::size_t i = 0; i < rows.size(); ++i) {
        out << rows[i].n << ','
            << rows[i].algorithm << ','
            << rows[i].run_idx << ','
            << rows[i].ms << ','
            << (rows[i].extrapolated ? 1 : 0)
            << '\n';
    }
}

static void run_validate(std::size_t n, std::uint64_t seed) {
    std::vector<Point> points = generate_points(n, seed, -100000.0, 100000.0);

    ClosestPairResult brute;
    ClosestPairResult div;
    const double brute_ms = measure_bruteforce_ms(points, &brute);
    const double div_ms = measure_divide_ms(points, &div);

    const double brute_d = std::sqrt(brute.dist2);
    const double div_d = std::sqrt(div.dist2);

    std::cout << "[validate] n=" << n << " seed=" << seed << "\n";
    std::cout << "  brute force: distance=" << std::setprecision(15) << brute_d << " ms=" << std::setprecision(6) << brute_ms << "\n";
    std::cout << "  divide and conquer: distance=" << std::setprecision(15) << div_d << " ms=" << std::setprecision(6) << div_ms << "\n";
    std::cout << "  abs diff=" << std::setprecision(17) << std::abs(brute_d - div_d) << "\n";
}

static void run_export_points(std::size_t n, std::uint64_t seed, const std::string &points_csv, const std::string &pair_csv) {
    std::vector<Point> points = generate_points(n, seed, -1000.0, 1000.0);
    const ClosestPairResult res = divide_conquer(points);
    save_points_csv(points_csv, points);
    save_pair_csv(pair_csv, res, points);

    std::cout << "[export] points=" << points_csv << " pair=" << pair_csv << " n=" << n << "\n";
    std::cout << "  closest distance=" << std::setprecision(15) << std::sqrt(res.dist2) << "\n";
}

static void run_export_trace(
    std::size_t n,
    std::uint64_t seed,
    const std::string &points_csv,
    const std::string &pair_csv,
    const std::string &trace_csv
) {
    std::vector<Point> points = generate_points(n, seed, -1000.0, 1000.0);
    std::vector<TraceEvent> events;
    const ClosestPairResult res = divide_conquer_with_trace(points, &events);

    save_points_csv(points_csv, points);
    save_pair_csv(pair_csv, res, points);
    save_trace_csv(trace_csv, events);

    std::cout << "[export-trace] points=" << points_csv << " pair=" << pair_csv << " trace=" << trace_csv << " n=" << n << "\n";
    std::cout << "  closest distance=" << std::setprecision(15) << std::sqrt(res.dist2) << " updates=" << events.size() << "\n";
}

static ClosestPairResult divide_conquer_recursive_with_steps(
    const std::vector<Point> &points_by_x,
    const std::vector<int> &indices_sorted_by_y,
    int left,
    int right,
    int depth,
    std::vector<TraceEvent> &events,
    std::vector<DivisionStep> &steps,
    int &step_counter
) {
    const int count = right - left;
    if (count <= 3) {
        ClosestPairResult best;
        best.a = -1;
        best.b = -1;
        best.dist2 = std::numeric_limits<double>::infinity();

        for (int i = left; i < right; ++i) {
            for (int j = i + 1; j < right; ++j) {
                const double d2 = distance2(points_by_x[i], points_by_x[j]);
                if (d2 < best.dist2) {
                    best.a = points_by_x[i].id;
                    best.b = points_by_x[j].id;
                    best.dist2 = d2;

                    TraceEvent e;
                    e.step = ++step_counter;
                    e.id_a = best.a;
                    e.id_b = best.b;
                    e.dist2 = best.dist2;
                    e.phase = "base";
                    events.push_back(e);
                }
            }
        }
        return best;
    }

    const int mid = left + count / 2;
    const double mid_x = points_by_x[mid].x;

    std::vector<int> left_y;
    std::vector<int> right_y;
    left_y.reserve(indices_sorted_by_y.size());
    right_y.reserve(indices_sorted_by_y.size());

    for (std::size_t i = 0; i < indices_sorted_by_y.size(); ++i) {
        const int idx = indices_sorted_by_y[i];
        if (idx >= left && idx < mid) {
            left_y.push_back(idx);
        } else if (idx >= mid && idx < right) {
            right_y.push_back(idx);
        }
    }

    const ClosestPairResult left_best = divide_conquer_recursive_with_steps(points_by_x, left_y, left, mid, depth + 1, events, steps, step_counter);
    const ClosestPairResult right_best = divide_conquer_recursive_with_steps(points_by_x, right_y, mid, right, depth + 1, events, steps, step_counter);
    ClosestPairResult best = better(left_best, right_best);

    std::vector<int> strip;
    strip.reserve(indices_sorted_by_y.size());
    for (std::size_t i = 0; i < indices_sorted_by_y.size(); ++i) {
        const int idx = indices_sorted_by_y[i];
        const double dx = points_by_x[idx].x - mid_x;
        if (dx * dx < best.dist2) {
            strip.push_back(idx);
        }
    }

    DivisionStep div_step;
    div_step.depth = depth;
    div_step.left = left;
    div_step.right = right;
    div_step.mid_x = mid_x;
    div_step.strip_left = mid_x - std::sqrt(best.dist2);
    div_step.strip_right = mid_x + std::sqrt(best.dist2);
    div_step.best_dist = std::sqrt(best.dist2);
    steps.push_back(div_step);

    for (std::size_t i = 0; i < strip.size(); ++i) {
        for (std::size_t j = i + 1; j < strip.size(); ++j) {
            const double dy = points_by_x[strip[j]].y - points_by_x[strip[i]].y;
            if (dy * dy >= best.dist2) {
                break;
            }
            const double d2 = distance2(points_by_x[strip[i]], points_by_x[strip[j]]);
            if (d2 < best.dist2) {
                best.a = points_by_x[strip[i]].id;
                best.b = points_by_x[strip[j]].id;
                best.dist2 = d2;

                TraceEvent e;
                e.step = ++step_counter;
                e.id_a = best.a;
                e.id_b = best.b;
                e.dist2 = best.dist2;
                e.phase = "strip";
                events.push_back(e);
            }
        }
    }

    return best;
}

static ClosestPairResult divide_conquer_with_steps(
    std::vector<Point> points,
    std::vector<TraceEvent> *events_out,
    std::vector<DivisionStep> *steps_out
) {
    if (points.size() < 2) {
        ClosestPairResult r;
        r.a = -1;
        r.b = -1;
        r.dist2 = std::numeric_limits<double>::infinity();
        return r;
    }

    std::sort(points.begin(), points.end(), [](const Point &lhs, const Point &rhs) {
        if (lhs.x != rhs.x) {
            return lhs.x < rhs.x;
        }
        if (lhs.y != rhs.y) {
            return lhs.y < rhs.y;
        }
        return lhs.id < rhs.id;
    });

    std::vector<int> by_y(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        by_y[i] = static_cast<int>(i);
    }
    std::sort(by_y.begin(), by_y.end(), [&](int a, int b) {
        if (points[a].y != points[b].y) {
            return points[a].y < points[b].y;
        }
        if (points[a].x != points[b].x) {
            return points[a].x < points[b].x;
        }
        return points[a].id < points[b].id;
    });

    std::vector<TraceEvent> local_events;
    std::vector<DivisionStep> local_steps;
    int step_counter = 0;
    ClosestPairResult res = divide_conquer_recursive_with_steps(points, by_y, 0, static_cast<int>(points.size()), 0, local_events, local_steps, step_counter);

    if (events_out != nullptr) {
        *events_out = local_events;
    }
    if (steps_out != nullptr) {
        *steps_out = local_steps;
    }
    return res;
}

static void run_export_divcon(
    std::size_t n,
    std::uint64_t seed,
    const std::string &points_csv,
    const std::string &pair_csv,
    const std::string &trace_csv,
    const std::string &steps_csv
) {
    std::vector<Point> points = generate_points(n, seed, -1000.0, 1000.0);
    std::vector<TraceEvent> events;
    std::vector<DivisionStep> steps;
    const ClosestPairResult res = divide_conquer_with_steps(points, &events, &steps);

    save_points_csv(points_csv, points);
    save_pair_csv(pair_csv, res, points);
    save_trace_csv(trace_csv, events);
    save_division_csv(steps_csv, steps);

    std::cout << "[export-divcon] points=" << points_csv << " pair=" << pair_csv << " trace=" << trace_csv << " steps=" << steps_csv << " n=" << n << "\n";
    std::cout << "  closest distance=" << std::setprecision(15) << std::sqrt(res.dist2) << " updates=" << events.size() << " division_steps=" << steps.size() << "\n";
}

static void run_benchmark(
    std::size_t n_start,
    std::size_t n_end,
    std::size_t step,
    int repeats,
    std::size_t brute_limit,
    std::uint64_t seed,
    const std::string &output_csv
) {
    std::vector<BenchRow> rows;

    std::cout << "[benchmark] n=" << n_start << ".." << n_end << " step=" << step
              << " repeats=" << repeats << " brute_limit=" << brute_limit << "\n";

    // Always measure brute-force at brute_limit as a baseline for O(n^2) extrapolation.
    if (brute_limit >= 2) {
        std::cout << "  running brute-force baseline at n=" << brute_limit << " ...\n";
        for (int r = 0; r < repeats; ++r) {
            const std::uint64_t current_seed = seed + static_cast<std::uint64_t>(brute_limit) * 977 + static_cast<std::uint64_t>(r);
            std::vector<Point> base_points = generate_points(brute_limit, current_seed, -1000000.0, 1000000.0);

            BenchRow brute_base;
            brute_base.n = brute_limit;
            brute_base.algorithm = "bruteforce";
            brute_base.run_idx = r + 1;
            brute_base.ms = measure_bruteforce_ms(base_points, nullptr);
            brute_base.extrapolated = false;
            rows.push_back(brute_base);
        }
    }

    for (std::size_t n = n_start; n <= n_end; n += step) {
        std::cout << "  running n=" << n << " ...\n";
        for (int r = 0; r < repeats; ++r) {
            const std::uint64_t current_seed = seed + static_cast<std::uint64_t>(n) * 131 + static_cast<std::uint64_t>(r);
            std::vector<Point> points = generate_points(n, current_seed, -1000000.0, 1000000.0);

            BenchRow div_row;
            div_row.n = n;
            div_row.algorithm = "divide";
            div_row.run_idx = r + 1;
            div_row.ms = measure_divide_ms(points, nullptr);
            div_row.extrapolated = false;
            rows.push_back(div_row);

            if (n <= brute_limit && n != brute_limit) {
                BenchRow brute_row;
                brute_row.n = n;
                brute_row.algorithm = "bruteforce";
                brute_row.run_idx = r + 1;
                brute_row.ms = measure_bruteforce_ms(points, nullptr);
                brute_row.extrapolated = false;
                rows.push_back(brute_row);
            }
        }

        if (step == 0) {
            break;
        }
        if (n_end - n < step) {
            break;
        }
    }

    // Use the largest measured brute-force point to extrapolate larger scales by O(n^2).
    std::size_t base_n = 0;
    double base_ms = 0.0;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        if (rows[i].algorithm == "bruteforce" && !rows[i].extrapolated) {
            if (rows[i].n > base_n) {
                base_n = rows[i].n;
                base_ms = rows[i].ms;
            }
        }
    }

    if (base_n > 0 && base_ms > 0.0) {
        for (std::size_t n = n_start; n <= n_end; n += step) {
            if (n > brute_limit) {
                BenchRow ext_row;
                ext_row.n = n;
                ext_row.algorithm = "bruteforce";
                ext_row.run_idx = 0;
                const double ratio = static_cast<double>(n) / static_cast<double>(base_n);
                ext_row.ms = base_ms * ratio * ratio;
                ext_row.extrapolated = true;
                rows.push_back(ext_row);
            }
            if (step == 0) {
                break;
            }
            if (n_end - n < step) {
                break;
            }
        }
    }

    write_benchmark_csv(output_csv, rows);
    std::cout << "[benchmark] done. output=" << output_csv << " rows=" << rows.size() << "\n";
}

int main(int argc, char **argv) {
    std::vector<std::string> args;
    args.reserve(static_cast<std::size_t>(argc));
    for (int i = 1; i < argc; ++i) {
        args.push_back(std::string(argv[i]));
    }

    const std::string mode = get_arg_value(args, "--mode", "validate");

    if (mode == "validate") {
        const std::size_t n = static_cast<std::size_t>(std::stoull(get_arg_value(args, "--n", "2000")));
        const std::uint64_t seed = static_cast<std::uint64_t>(std::stoull(get_arg_value(args, "--seed", "20260416")));
        run_validate(n, seed);
        return 0;
    }

    if (mode == "export") {
        const std::size_t n = static_cast<std::size_t>(std::stoull(get_arg_value(args, "--n", "200")));
        const std::uint64_t seed = static_cast<std::uint64_t>(std::stoull(get_arg_value(args, "--seed", "20260416")));
        const std::string points_csv = get_arg_value(args, "--points-csv", "../results/points_demo.csv");
        const std::string pair_csv = get_arg_value(args, "--pair-csv", "../results/pair_demo.csv");
        run_export_points(n, seed, points_csv, pair_csv);
        return 0;
    }

    if (mode == "benchmark") {
        const std::size_t n_start = static_cast<std::size_t>(std::stoull(get_arg_value(args, "--n-start", "100000")));
        const std::size_t n_end = static_cast<std::size_t>(std::stoull(get_arg_value(args, "--n-end", "1000000")));
        const std::size_t step = static_cast<std::size_t>(std::stoull(get_arg_value(args, "--step", "100000")));
        const int repeats = std::stoi(get_arg_value(args, "--repeats", "1"));
        const std::size_t brute_limit = static_cast<std::size_t>(std::stoull(get_arg_value(args, "--brute-limit", "20000")));
        const std::uint64_t seed = static_cast<std::uint64_t>(std::stoull(get_arg_value(args, "--seed", "20260416")));
        const std::string output_csv = get_arg_value(args, "--output", "../results/benchmark.csv");

        run_benchmark(n_start, n_end, step, repeats, brute_limit, seed, output_csv);
        return 0;
    }

    if (mode == "export-trace") {
        const std::size_t n = static_cast<std::size_t>(std::stoull(get_arg_value(args, "--n", "200")));
        const std::uint64_t seed = static_cast<std::uint64_t>(std::stoull(get_arg_value(args, "--seed", "20260416")));
        const std::string points_csv = get_arg_value(args, "--points-csv", "../results/points_demo.csv");
        const std::string pair_csv = get_arg_value(args, "--pair-csv", "../results/pair_demo.csv");
        const std::string trace_csv = get_arg_value(args, "--trace-csv", "../results/trace_demo.csv");

        run_export_trace(n, seed, points_csv, pair_csv, trace_csv);
        return 0;
    }

    if (mode == "export-divcon") {
        const std::size_t n = static_cast<std::size_t>(std::stoull(get_arg_value(args, "--n", "200")));
        const std::uint64_t seed = static_cast<std::uint64_t>(std::stoull(get_arg_value(args, "--seed", "20260416")));
        const std::string points_csv = get_arg_value(args, "--points-csv", "../results/points_demo.csv");
        const std::string pair_csv = get_arg_value(args, "--pair-csv", "../results/pair_demo.csv");
        const std::string trace_csv = get_arg_value(args, "--trace-csv", "../results/trace_demo.csv");
        const std::string steps_csv = get_arg_value(args, "--steps-csv", "../results/steps_demo.csv");

        run_export_divcon(n, seed, points_csv, pair_csv, trace_csv, steps_csv);
        return 0;
    }

    std::cerr << "Unknown mode: " << mode << "\n";
    std::cerr << "Usage:\n";
    std::cerr << "  --mode validate --n 2000 --seed 20260416\n";
    std::cerr << "  --mode export --n 200 --points-csv ../results/points_demo.csv --pair-csv ../results/pair_demo.csv\n";
    std::cerr << "  --mode export-trace --n 200 --points-csv ../results/points_demo.csv --pair-csv ../results/pair_demo.csv --trace-csv ../results/trace_demo.csv\n";
    std::cerr << "  --mode export-divcon --n 200 --points-csv ../results/points_demo.csv --pair-csv ../results/pair_demo.csv --trace-csv ../results/trace_demo.csv --steps-csv ../results/steps_demo.csv\n";
    std::cerr << "  --mode benchmark --n-start 100000 --n-end 1000000 --step 100000 --repeats 1 --brute-limit 20000 --output ../results/benchmark.csv\n";

    if (has_flag(args, "--help") || has_flag(args, "-h")) {
        return 0;
    }
    return 1;
}
