#include <cstdio>

#include <chrono>
#include <future>
#include <latch>
#include <mutex>
#include <numeric>
#include <thread>

#include <fmt/core.h>
#include <tinyopt/tinyopt.h>

#include "util.h"

using value_type = double;

const char* usage_str =
    "[OPTION]...\n"
    "\n"
    "  -g, --gpu=bench      Perform benchmark on gpu, one of: none, gemm, stream\n"
    "  -c, --cpu=bench      Perform benchmark on cpu, one of: none, gemm, stream\n"
    "  -d, --duration=N     duration in N seconds\n"
    "  -h, --help           Display usage information and exit\n";


std::latch work_wait_latch(3);

template <class... Args>
void print_safe(fmt::format_string<Args...> s, Args&&... args) {
    static std::mutex stdout_guard;
    std::lock_guard<std::mutex> _(stdout_guard);
    fmt::print(s, std::forward<Args>(args)...);
    std::fflush(stdout);
}

// there are three workloads that can be run:
//  none:    do nothing
//  dgemm:   run dgemm kernels full speed
//  stream:  run dgemm kernels full speed
enum class benchmark_kind {none, gemm, stream};
const char* benchmark_string(benchmark_kind k) {
    if      (k==benchmark_kind::none)   return "none";
    else if (k==benchmark_kind::gemm)   return "gemm";

    return "stream";
}

struct config {
    benchmark_kind gpu = benchmark_kind::gemm;
    benchmark_kind cpu = benchmark_kind::none;
    uint32_t duration = 10;      // seconds
};

void gpu_work(config);
void cpu_work(config);

int main(int argc, char** argv) {
    config cfg;

    try {
        using namespace to::literals;

        auto help = [argv0 = argv[0]] { to::usage(argv0, usage_str); };

        std::pair<const char*, benchmark_kind> functions[] = {
            {"none",   benchmark_kind::none},
            {"gemm",   benchmark_kind::gemm},
            {"stream", benchmark_kind::stream},
        };

        to::option opts[] = {
            {{cfg.gpu, to::keywords(functions)}, "-g"_compact, "--gpu"},
            {{cfg.cpu, to::keywords(functions)}, "-c"_compact, "--cpu"},
            {cfg.duration, "-d"_compact, "--duration"},
            {to::action(help), to::flag, to::exit, "-h", "--help"},
        };

        if (!to::run(opts, argc, argv+1)) return 0;

        if (argv[1]) throw to::option_error("unrecogonized argument", argv[1]);
    }
    catch (to::option_error& e) {
        to::usage_error(argv[0], usage_str, e.what());
        return 1;
    }

    print_safe("--- Node-Burn ---\n");
    print_safe("  gpu: {}\n", benchmark_string(cfg.gpu));
    print_safe("  cpu: {}\n\n", benchmark_string(cfg.cpu));

    auto gpu_handle = std::async(std::launch::async, gpu_work, cfg);
    auto cpu_handle = std::async(std::launch::async, cpu_work, cfg);
    work_wait_latch.arrive_and_wait();

    print_safe("--- burning for {} seconds\n", cfg.duration);

    gpu_handle.wait();
    cpu_handle.wait();

    print_safe("finished\n");
}

std::string flop_report(uint32_t N, std::vector<double> times) {
    std::sort(times.begin(), times.end());
    double duration = std::accumulate(times.begin(), times.end(), 0.);
    auto runs = times.size();
    size_t flops_per_mul = 2*N*N*N;
    size_t flops_total = runs*flops_per_mul;
    size_t bytes = N*N*3*sizeof(value_type);

    double gflops = 1e-9 * flops_total / duration;

    //auto flops = [flops_per_mul] (double t) -> int {return int(std::round(flops_per_mul/t*1e-9));};

    return fmt::format("{:8d} gemm {:8.2f} GFlops {:8.3f} seconds {:8.3f} Gbytes", runs, gflops, duration, 1e-9*bytes);
}

void gpu_work(config cfg) {
    using namespace std::chrono_literals;

    auto start_init = timestamp();

    // INITIALISE
    const std::uint64_t N = 32000;
    const value_type alpha = 0.99;
    const value_type beta = 1./(N*N);

    auto a = malloc_device<value_type>(N*N);
    auto b = malloc_device<value_type>(N*N);
    auto c = malloc_device<value_type>(N*N);

    gpu_rand(a, N*N);
    gpu_rand(b, N*N);
    gpu_rand(c, N*N);

    // call once
    gpu_gemm(a, b, c, N, N, N, alpha, beta);

    device_synchronize();

    // synchronise before burning
    print_safe("gpu: finished intialisation in {} seconds\n", duration(start_init));
    work_wait_latch.arrive_and_wait();

    std::vector<double> times;
    auto start_fire = timestamp();
    while (duration(start_fire)<cfg.duration) {
        device_synchronize();
        auto start = timestamp();
        gpu_gemm(a, b, c, N, N, N, alpha, beta);
        device_synchronize();
        auto stop = timestamp();
        times.push_back(duration(start, stop));
    }

    print_safe("gpu: {}\n", flop_report(N, times));
}

void cpu_work(config cfg) {
    using namespace std::chrono_literals;
    if (cfg.cpu==benchmark_kind::gemm) {
        auto start_init = timestamp();

        // INITIALISE
        const std::uint64_t N = 1000;
        const value_type alpha = 0.99;
        const value_type beta = 1./(N*N);

        auto a = malloc_host<value_type>(N*N);
        auto b = malloc_host<value_type>(N*N);
        auto c = malloc_host<value_type>(N*N);

        cpu_rand(a, N*N);
        cpu_rand(b, N*N);
        cpu_rand(c, N*N);

        // call once
        cpu_gemm(a, b, c, N, N, N, alpha, beta);

        // synchronise before burning
        print_safe("cpu: finished intialisation in {} seconds\n", duration(start_init));
        work_wait_latch.arrive_and_wait();

        std::vector<double> times;
        auto start_fire = timestamp();
        while (duration(start_fire)<cfg.duration) {
            auto start = timestamp();
            cpu_gemm(a, b, c, N, N, N, alpha, beta);
            auto stop = timestamp();
            times.push_back(duration(start, stop));
        }

        print_safe("cpu: {}\n", flop_report(N, times));
    }
    else {
        work_wait_latch.arrive_and_wait();
        print_safe("cpu: no work\n");
    }
}
