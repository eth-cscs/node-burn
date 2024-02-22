#include <cstdint>
#include <memory>
#include <iostream>
#include <type_traits>

#include "experiment.h"

extern "C" void dgemm_(char*, char*, int*, int*,int*, double*, double*, int*, double*, int*, double*, double*, int*);
extern "C" void sgemm_(char*, char*, int*, int*,int*, float*, float*, int*, float*, int*, float*, float*, int*);

template <typename T>
T* malloc_host(size_t N, T value=T()) {
    T* ptr = (T*)(malloc(N*sizeof(T)));
    std::fill(ptr, ptr+N, value);

    return ptr;
}

template<class T>
void cpu_gemm(T* a, T*b, T*c,
          int m, int n, int k,
          T alpha, T beta)
{
    char trans = 'N';

    if constexpr (std::is_same_v<T, double>) {
        dgemm_(&trans, &trans, &m, &n, &k, &alpha, a, &m, b, &k, &beta, c, &m);
    }
    else {
        sgemm_(&trans, &trans, &m, &n, &k, &alpha, a, &m, b, &k, &beta, c, &m);
    }
}

// TODO: this should take additional arguments for the matrix dimensions, and
// use the i,j values as 32 bit inputs to the hashing generator.
template<class T>
void cpu_rand(T* x, uint64_t n) {
    auto xorshiftstar = [](uint64_t x) -> uint64_t {
        x ^= x >> 12; // a
        x ^= x << 25; // b
        x ^= x >> 27; // c
        return x * 0x2545F4914F6CDD1D;
    };

    auto generate_random_double = [&xorshiftstar](uint32_t u1, uint32_t u2) -> T {
        uint64_t combined = ((uint64_t)u1 << 32) | u2;
        uint64_t hashed = xorshiftstar(combined);
        return (T)hashed / (T)UINT64_MAX;
    };

    for (std::size_t i=0; i<n; ++i) {
        x[i] = generate_random_double(0, n) - 0.5;
    }
}

template<class T>
void cpu_stream_triad(T* __restrict__ a, T* __restrict__ b, T* __restrict__ c, T scale, std::uint64_t N)
{
    #pragma omp parallel for schedule(static)
    for (std::uint64_t i = 0; i < N; ++i)
    {
        a[i] = b[i] + c[i] * scale;
    }
}

template <typename T>
struct cpu_gemm_state: public benchmark {
    using value_type = T;

    cpu_gemm_state(std::uint32_t N):
        benchmark(benchmark_kind::gemm),
        N(N),
        beta(1./(N*N))
    {}

    void init() {
        a = malloc_host<value_type>(N*N);
        b = malloc_host<value_type>(N*N);
        c = malloc_host<value_type>(N*N);
        cpu_rand(a, N*N);
        cpu_rand(b, N*N);
        cpu_rand(c, N*N);
    }

    void run() {
        cpu_gemm(a, b, c, N, N, N, alpha, beta);
    }

    void synchronize() {}

    std::string report(std::vector<double> times) {
        return flop_report_gemm(N, std::move(times));
    }

    ~cpu_gemm_state() {
        free(a);
        free(b);
        free(c);
    }

private:
    const std::uint32_t N;
    const value_type alpha = 0.99;
    const value_type beta;

    value_type* a;
    value_type* b;
    value_type* c;
};

template <typename T>
struct cpu_stream_state: public benchmark {
    using value_type = T;

    cpu_stream_state(std::uint32_t N):
        benchmark(benchmark_kind::stream),
        N(N)
    {}

    void init() {
        a = malloc_host<value_type>(N);
        b = malloc_host<value_type>(N);
        c = malloc_host<value_type>(N);
        cpu_rand(a, N);
        cpu_rand(b, N);
        cpu_rand(c, N);
    }

    void run() {
        cpu_stream_triad(a, b, c, alpha, N);
    }

    void synchronize() {}

    std::string report(std::vector<double> times) {
        return bandwidth_report_stream(N, std::move(times));
    }

    ~cpu_stream_state() {
        free(a);
        free(b);
        free(c);
    }

private:
    const std::uint32_t N;
    const value_type alpha = 0.99;

    value_type* a;
    value_type* b;
    value_type* c;
};

std::unique_ptr<benchmark> get_cpu_benchmark(const experiment& e) {
    switch (e.kind) {
        case benchmark_kind::gemm:
            return std::make_unique<cpu_gemm_state<value_type>>(e.args[0]);
        case benchmark_kind::stream:
            return std::make_unique<cpu_stream_state<value_type>>(e.args[0]);
        default:
            return std::make_unique<null_benchmark>();
    }
}
