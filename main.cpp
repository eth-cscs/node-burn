#include <cstdio>

#include <barrier>
#include <chrono>
#include <future>
#include <mutex>
#include <thread>

#include <unistd.h>

#include <fmt/core.h>
#include <tinyopt/tinyopt.h>

#include "experiment.h"
#include "instrument.h"
#include "timers.h"

#ifdef NB_INSTRUMENT
constexpr bool with_instrument = true;
#else
constexpr bool with_instrument = false;
#endif

using value_type = double;

const char* usage_str =
    "[OPTION]...\n"
    "\n"
    "  -g, --gpu=bench      Perform benchmark on gpu, one of: none, gemm, stream\n"
    "  -c, --cpu=bench      Perform benchmark on cpu, one of: none, gemm, stream\n"
    "  -d, --duration=N     duration in N seconds\n"
    "  -b, --batch          enable batch mode (less verbose output)\n"
    "  -h, --help           Display usage information and exit\n";

// There is a worker thread for each hardware target
constexpr int num_workers = with_gpu? 2: 1;

std::barrier B(num_workers+1);
bool batch_mode = false;

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
    // duration of the benchmark in seconds
    uint32_t duration = 10;
};

void run_work(std::unique_ptr<benchmark> state, std::string prefix, std::uint32_t total_duration);

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
            {to::set(batch_mode, true), to::flag, "-b", "--batch"},
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

    if (!batch_mode) {
        print_safe("--------- Node-Burn ---------\n");
        print_safe("experiments:\n");
        print_safe("  gpu: {}\n", cfg.gpu);
        print_safe("  cpu: {}\n", cfg.cpu);
        print_safe("  duration: {} seconds\n", cfg.duration);
        print_safe("-----------------------------\n\n");
    }

    if (cfg.gpu.kind!=benchmark_kind::none && !with_gpu) {
        print_safe("WARNING: GPU is not enabled. Recompile with GPU enabled to burn the GPU.\n");
        exit(1);
    }

    using job_handle = decltype (std::async(std::launch::async, [](){return;}));
    std::vector<job_handle> jobs;

    auto instrument = get_instrument();

    if (with_gpu) {
        jobs.push_back(
                std::async(
                    std::launch::async,
                    run_work,
                    get_gpu_benchmark(cfg.gpu), "gpu", cfg.duration));
    }

    jobs.push_back(
            std::async(
                std::launch::async,
                run_work,
                get_cpu_benchmark(cfg.cpu), "cpu", cfg.duration));

    // all tasks wait after starting initialisation
    B.arrive_and_wait();
    instrument->start();
    // START
    // all tasks start workload
    if (!batch_mode) print_safe("\n--- burning for {} seconds\n\n", cfg.duration);
    // all tasks wait after finishing work
    B.arrive_and_wait();
    instrument->stop();
    // STOP

    for (auto& job: jobs) job.wait();

    char host[512];
    gethostname(host, 511);
    instrument->print_result(fmt::format("{}:", host));

    return 0;
}

void run_work(std::unique_ptr<benchmark> state, std::string prefix, std::uint32_t total_duration) {
    using namespace std::chrono_literals;

    std::vector<double> times;
    times.reserve(100000);
    auto start_init = timestamp();

    char host[512];
    gethostname(host, 511);

    std::string full_prefix = host + (":" + prefix);
    // intialise state and run once
    state->init();
    state->run();

    // synch before burning
    state->synchronize();
    if (!batch_mode) print_safe("{} finished initialisation in {} seconds\n", full_prefix, duration(start_init));
    B.arrive_and_wait();

    auto start_fire = timestamp();
    while (duration(start_fire)<total_duration) {
        auto start = timestamp();
        state->run();
        state->synchronize();
        auto stop = timestamp();
        times.push_back(duration(start, stop));
    }

    B.arrive_and_wait();
    if (!batch_mode || state->kind!=benchmark_kind::none)
    print_safe("{} {}\n", full_prefix, state->report(times));
}
