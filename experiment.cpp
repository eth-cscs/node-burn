#include <algorithm>
#include <numeric>

#include <fmt/core.h>

#include "experiment.h"

std::vector<std::string> split(const std::string_view v, const char delim=',') {
    std::vector<std::string> results;

    auto pos = v.begin();
    auto end = v.end();
    auto next = std::find(pos, end, delim);
    while (next!=end) {
        results.emplace_back(pos, next);
        pos = next+1;
        next = std::find(pos, end, delim);
    }
    results.emplace_back(pos, next);

    return results;
}

experiment::experiment(std::string_view s) {
    auto results = split(s, ',');
    auto pos = results.begin();
    std::string name = *pos;
    if (name=="none" || name=="") {
        kind = benchmark_kind::none;
    }
    else if (name=="gemm") {
        kind = benchmark_kind::gemm;
    }
    else if (name=="stream") {
        kind = benchmark_kind::stream;
    }
    else {
        throw std::invalid_argument(
            fmt::format(
                "invalid benchmark kind '{}' in '{}': pick one of gemm or none",
                name, s));
    }
    while (++pos!=results.end()) {
        try {
            args.push_back(std::stoi(*pos));
        }
        catch (std::exception &e) {
            throw std::invalid_argument(
                fmt::format(
                    "invalid argument '{}' in '{}': expected an integer",
                    *pos, s));
        }
    }

    // set defaults
    if (kind == benchmark_kind::gemm) {
        if (args.empty()) {
            args.push_back(4000);
        }
    }
    if (kind == benchmark_kind::stream) {
        if (args.empty()) {
            args.push_back(500000000);
        }
    }
}

std::string flop_report_gemm(uint32_t N, std::vector<double> times) {
    std::sort(times.begin(), times.end());
    double duration = std::accumulate(times.begin(), times.end(), 0.);
    auto runs = times.size();
    double flops_per_mul = 2.0*N*N*N;
    double flops_total = runs*flops_per_mul;
    size_t bytes = N*N*3*sizeof(value_type);

    double gflops = 1e-9 * flops_total / duration;

    return fmt::format("{:6d} iterations, {:8.2F} GFlops, {:8.1F} seconds, {:8.3F} Gbytes", runs, gflops, duration, 1e-9*bytes);
}

std::string bandwidth_report_stream(uint64_t N, std::vector<double> times) {
    std::sort(times.begin(), times.end());
    double duration = std::accumulate(times.begin(), times.end(), 0.);
    auto runs = times.size();
    double bytes_per_call = 3.0 * sizeof(value_type) * N;
    double bytes_total = runs * bytes_per_call;

    double GB_per_second = 1e-9 * bytes_total / duration;

    return fmt::format("{:6d} iterations, {:8.2F} GB/s, {:8.1F} seconds", runs, GB_per_second, duration);
}

