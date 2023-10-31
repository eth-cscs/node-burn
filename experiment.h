#pragma once

#include <string>
#include <string_view>
#include <vector>

#include <fmt/core.h>

// there are three workloads that can be run:
//  none:    do nothing
//  dgemm:   run dgemm kernels full speed
//  stream:  run dgemm kernels full speed
enum class benchmark_kind {none, gemm, stream};


static const char* benchmark_string(benchmark_kind k) {
    if      (k==benchmark_kind::none)   return "none";
    else if (k==benchmark_kind::gemm)   return "gemm";

    return "stream";
}

struct experiment {
    experiment(std::string_view);
    experiment() = default;
    benchmark_kind kind = benchmark_kind::none;
    std::vector<uint64_t> args;
};

// fmt library gubbins.

template<>
struct fmt::formatter<experiment>
{
    template<typename ParseContext>
    constexpr auto parse(ParseContext& ctx);

    template<typename FormatContext>
    auto format(experiment const& e, FormatContext& ctx);
};

template<typename ParseContext>
constexpr auto fmt::formatter<experiment>::parse(ParseContext& ctx)
{
    return ctx.begin();
}

template<typename FormatContext>
auto fmt::formatter<experiment>::format(experiment const& e, FormatContext& ctx)
{
    if (e.kind==benchmark_kind::none) {
        return fmt::format_to(ctx.out(), "none");
    }
    else if (e.kind==benchmark_kind::gemm) {
        return fmt::format_to(ctx.out(), "gemm, N={}", e.args[0]);
    }
    return fmt::format_to(ctx.out(), "stream triad");
}
