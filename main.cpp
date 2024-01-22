#include <cstdio>

#include <chrono>
#include <future>
#include <latch>
#include <mutex>
#include <numeric>
#include <thread>

#include <fmt/core.h>
#include <tinyopt/tinyopt.h>

#include "experiment.h"
#include "numeric.h"
#include "stream_gpu.h"
#include "timers.h"
#include "util.h"

using value_type = double;

const char* usage_str =
    "[OPTION]...\n"
    "\n"
    "  -g, --gpu=bench      Perform benchmark on gpu, one of: none, gemm, stream\n"
    "  -c, --cpu=bench      Perform benchmark on cpu, one of: none, gemm, stream\n"
    "  -d, --duration=N     duration in N seconds\n"
    "  -d, --duration=N     duration in N seconds\n"
    "  -h, --help           Display usage information and exit\n";


std::latch work_wait_latch(3);
std::latch work_finish_latch(3);

template <class... Args>
void print_safe(fmt::format_string<Args...> s, Args&&... args) {
    static std::mutex stdout_guard;
    std::lock_guard<std::mutex> _(stdout_guard);
    fmt::print(s, std::forward<Args>(args)...);
    std::fflush(stdout);
}

struct config {
    experiment gpu;
    experiment cpu;
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

        std::string gpu="none";
        std::string cpu="none";
        to::option opts[] = {
            {gpu, "-g"_compact, "--gpu"},
            {cpu, "-c"_compact, "--cpu"},
            {cfg.duration, "-d"_compact, "--duration"},
            {to::action(help), to::flag, to::exit, "-h", "--help"},
        };

        if (!to::run(opts, argc, argv+1)) return 0;

        cfg.cpu = {cpu};
        cfg.gpu = {gpu};

        if (argv[1]) throw to::option_error("unrecogonized argument", argv[1]);
    }
    catch (to::option_error& e) {
        to::usage_error(argv[0], usage_str, e.what());
        return 1;
    }

    print_safe("--------- Node-Burn ---------\n");
    print_safe("experiments:\n");
    print_safe("  gpu: {}\n", cfg.gpu);
    print_safe("  cpu: {}\n", cfg.cpu);
    print_safe("  duration: {} seconds\n", cfg.duration);
    print_safe("-----------------------------\n\n");

    auto gpu_handle = std::async(std::launch::async, gpu_work, cfg);
    auto cpu_handle = std::async(std::launch::async, cpu_work, cfg);
    work_wait_latch.arrive_and_wait();

    print_safe("\n--- burning for {} seconds\n\n", cfg.duration);

    work_finish_latch.arrive_and_wait();

    gpu_handle.wait();
    cpu_handle.wait();

    return 0;
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

void gpu_work(config cfg) {
#ifdef USE_CUDA    
    using namespace std::chrono_literals;

    if (cfg.gpu.kind==benchmark_kind::gemm) {
        auto start_init = timestamp();

        // INITIALISE
        const std::uint32_t N = cfg.gpu.args[0];
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
        print_safe("gpu: finished initialisation in {} seconds\n", duration(start_init));
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

        work_finish_latch.arrive_and_wait();
        print_safe("gpu: {}\n", flop_report_gemm(N, times));
 
        cudaFree(a);
        cudaFree(b);
        cudaFree(c);
    } else if (cfg.gpu.kind == benchmark_kind::stream) {
        auto start_init = timestamp();

        // INITIALISE
        const std::uint64_t N = cfg.gpu.args[0];
        const value_type alpha = 0.99;

        auto a = malloc_device<value_type>(N);
        auto b = malloc_device<value_type>(N);
        auto c = malloc_device<value_type>(N);

        gpu_rand(a, N);
        gpu_rand(b, N);
        gpu_rand(c, N);

        // call once
        gpu_stream_triad(a, b, c, alpha, N);

        device_synchronize();

        // synchronise before burning
        print_safe("gpu: finished initialisation in {} seconds\n", duration(start_init));
        work_wait_latch.arrive_and_wait();

        std::vector<double> times;
        auto start_fire = timestamp();
        while (duration(start_fire) < cfg.duration) {
            device_synchronize();
            auto start = timestamp();
            gpu_stream_triad(a, b, c, alpha, N);
            device_synchronize();
            auto stop = timestamp();
            times.push_back(duration(start, stop));
        }

        work_finish_latch.arrive_and_wait();
        print_safe("gpu: {}\n", bandwidth_report_stream(N, times));

        cudaFree(a);
        cudaFree(b);
        cudaFree(c);
    } else {
        work_wait_latch.arrive_and_wait();
        work_finish_latch.arrive_and_wait();
        print_safe("gpu: no work\n");
    }
#else
    work_wait_latch.arrive_and_wait();
    work_finish_latch.arrive_and_wait();
    print_safe("gpu: no work\n");
#endif
}

void cpu_work(config cfg) {
    using namespace std::chrono_literals;
    if (cfg.cpu.kind==benchmark_kind::gemm) {
        auto start_init = timestamp();

        // INITIALISE
        const std::uint32_t N = cfg.cpu.args[0];
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

        work_finish_latch.arrive_and_wait();
        print_safe("cpu: {}\n", flop_report_gemm(N, times));
    } else if (cfg.cpu.kind == benchmark_kind::stream) {
        auto start_init = timestamp();

        // INITIALISE
        const std::uint64_t N = cfg.cpu.args[0];
        const value_type alpha = 0.99;

        auto a = malloc_host<value_type>(N);
        auto b = malloc_host<value_type>(N);
        auto c = malloc_host<value_type>(N);

        cpu_rand(a, N);
        cpu_rand(b, N);
        cpu_rand(c, N);

        // call once
        cpu_stream_triad(a, b, c, alpha, N);

        // synchronise before burning
        print_safe("cpu: finished intialisation in {} seconds\n",
                   duration(start_init));
        work_wait_latch.arrive_and_wait();

        std::vector<double> times;
        auto start_fire = timestamp();
        while (duration(start_fire) < cfg.duration) {
            auto start = timestamp();
            cpu_stream_triad(a, b, c, alpha, N);
            auto stop = timestamp();
            times.push_back(duration(start, stop));
        }

        work_finish_latch.arrive_and_wait();
        print_safe("cpu: {}\n", bandwidth_report_stream(N, times));
    } else {
        work_wait_latch.arrive_and_wait();
        work_finish_latch.arrive_and_wait();
        print_safe("cpu: no work\n");
    }
}
