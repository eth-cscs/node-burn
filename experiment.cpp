#include <algorithm>

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
    if (kind==benchmark_kind::gemm) {
        if (args.empty()) {
            args.push_back(4000);
        }
    }
}

