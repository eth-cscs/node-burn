#include <cstdio>

#include <chrono>
#include <future>
#include <latch>
#include <mutex>
#include <thread>

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


std::mutex stdout_guard;
std::latch work_wait_latch(3);

void print(const std::string s) {
    std::lock_guard<std::mutex> _(stdout_guard);
    std::cout << s << std::endl;
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
        auto help = [argv0 = argv[0]] { to::usage(argv0, usage_str); };

        std::pair<const char*, benchmark_kind> functions[] = {
            {"none",   benchmark_kind::none},
            {"gemm",   benchmark_kind::gemm},
            {"stream", benchmark_kind::stream},
        };

        to::option opts[] = {
            {{cfg.gpu, to::keywords(functions)}, "-g", "--gpu"},
            {{cfg.cpu, to::keywords(functions)}, "-c", "--cpu"},
            {to::action(help), to::flag, to::exit, "-h", "--help"},
        };

        if (!to::run(opts, argc, argv+1)) return 0;

        if (argv[1]) throw to::option_error("unrecogonized argument", argv[1]);
    }
    catch (to::option_error& e) {
        to::usage_error(argv[0], usage_str, e.what());
        return 1;
    }

    std::printf("--- Node-Burn ---\n");
    std::printf("fire report:\n");
    std::printf("  gpu: %s\n", benchmark_string(cfg.gpu));
    std::printf("  cpu: %s\n", benchmark_string(cfg.cpu));
    std::printf("\n");
    std::fflush(stdout);

    auto gpu_handle = std::async(std::launch::async, gpu_work, cfg);
    print("main: gpu launched");
    auto cpu_handle = std::async(std::launch::async, cpu_work, cfg);
    print("main: cpu launched");
    work_wait_latch.arrive_and_wait();

    gpu_handle.wait();
    print("main: gpu finished");
    cpu_handle.wait();
    print("main: cpu finished");
}

void gpu_work(config cfg) {
    using namespace std::chrono_literals;

    // INITIALISE
    const std::uint64_t N = 10000;
    const value_type alpha = 0.99;
    const value_type beta = 1./(N*N);

    auto a = malloc_device<value_type>(N*N);
    auto b = malloc_device<value_type>(N*N);
    auto c = malloc_device<value_type>(N*N);

    gpu_rand(a, N*N);
    gpu_rand(b, N*N);
    gpu_rand(c, N*N);

    // call once
    gemm(a, b, c, N, N, N, alpha, beta);

    device_synchronize();

    // synchronise before burning
    print("gpu waiting");
    work_wait_latch.arrive_and_wait();

    std::vector<double> times;
    auto startloop = timestamp();
    while (duration(startloop)<cfg.duration) {
        device_synchronize();
        auto start = timestamp();
        gemm(a, b, c, N, N, N, alpha, beta);
        device_synchronize();
        auto stop = timestamp();
        times.push_back(duration(start, stop));
    }

    std::cout << "finished " << times.size() << " iterations in " << duration(startloop) << " seconds" << std::endl;

}

void cpu_work(config cfg) {
    using namespace std::chrono_literals;

    // perform initialisation
    std::this_thread::sleep_for(1000ms);

    // synchronise before burning
    print("cpu waiting");
    work_wait_latch.arrive_and_wait();
}
