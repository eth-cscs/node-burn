#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <fmt/core.h>

#ifdef NB_GPU
constexpr bool with_gpu = true;
#else
constexpr bool with_gpu = false;
#endif

using value_type=double;

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
    uint32_t duration;
};


struct benchmark {
    benchmark(benchmark_kind k): kind(k) {}
    benchmark() = default;
    virtual void run() = 0;
    virtual void init() = 0;
    virtual void synchronize() = 0;
    virtual std::string report(std::vector<double>) = 0;
    virtual ~benchmark() {};
    const benchmark_kind kind = benchmark_kind::none;
};

struct null_benchmark: benchmark {
    null_benchmark() {}
    void init() {};
    void run() {};
    void synchronize() {};
    std::string report(std::vector<double>) {return "no benchmark run";};
};

#ifdef NB_GPU
std::unique_ptr<benchmark> get_gpu_benchmark(const experiment&);
#else
static
std::unique_ptr<benchmark> get_gpu_benchmark(const experiment&) {
    return std::make_unique<null_benchmark>();
};
#endif
std::unique_ptr<benchmark> get_cpu_benchmark(const experiment&);

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

#ifdef NB_GPU
std::string flop_report_gemm(uint32_t N, std::vector<double> times);
std::string bandwidth_report_stream(uint64_t N, std::vector<double> times);
#else
std::string flop_report_gemm(uint32_t N, std::vector<double> times);
std::string bandwidth_report_stream(uint64_t N, std::vector<double> times);
#endif
