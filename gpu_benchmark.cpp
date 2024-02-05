#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>

#include <memory>
#include <iostream>
#include <type_traits>

#include "experiment.h"
#include "stream_gpu.h"

// error checking
void check_status(cudaError_t status) {
    if(status != cudaSuccess) {
        std::cerr << "error: CUDA API call : "
                  << cudaGetErrorString(status) << std::endl;
        std::exit(-1);
    }
}

void check_status(curandStatus_t status) {
    if(status != CURAND_STATUS_SUCCESS) {
        std::cerr << "error: CURAND" << std::endl;
        std::exit(-1);
    }
}

// allocate space on GPU for n instances of type T
template <typename T>
T* malloc_device(size_t n) {
    void* p;
    auto status = cudaMalloc(&p, n*sizeof(T));
    check_status(status);
    return (T*)p;
}

template <typename T>
T* malloc_host(size_t N, T value=T()) {
    T* ptr = (T*)(malloc(N*sizeof(T)));
    std::fill(ptr, ptr+N, value);

    return ptr;
}

// helper for initializing cublas
// not threadsafe: if we want to burn more than one GPU this will need to be reworked.
cublasHandle_t get_blas_handle() {
    static bool is_initialized = false;
    static cublasHandle_t cublas_handle;

    if(!is_initialized) {
        cublasCreate(&cublas_handle);
    }
    return cublas_handle;
}

template<class T>
void gpu_gemm(T* a, T* b, T* c,
          int m, int n, int k,
          T alpha, T beta)
{
    if constexpr (std::is_same_v<T, double>)
    {
        cublasDgemm(
                get_blas_handle(),
                CUBLAS_OP_N, CUBLAS_OP_N,
                m, n, k,
                &alpha,
                a, m,
                b, k,
                &beta,
                c, m
        );
    }
    else {
        cublasSgemm(
                get_blas_handle(),
                CUBLAS_OP_N, CUBLAS_OP_N,
                m, n, k,
                &alpha,
                a, m,
                b, k,
                &beta,
                c, m
        );
    }
}

/*
template void gpu_gemm(float* a, float* b, float* c,
          int m, int n, int k,
          float alpha, float beta);

template void gpu_gemm(double* a, double* b, double* c,
          int m, int n, int k,
          double alpha, double beta);
*/

template<class T>
void gpu_rand(T* x, uint64_t n) {
    curandGenerator_t gen;

    check_status(
            curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    check_status(
            curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
    if constexpr (std::is_same_v<T, double>)
    {
        check_status(
                curandGenerateNormalDouble(gen, x, n, 0., 1.));
    }
    else
    {
        check_status(
                curandGenerateNormal(gen, x, n, 0., 1.));
    }
    check_status(
            curandDestroyGenerator(gen));
}


template <typename T>
struct gpu_gemm_state: public benchmark {
    using value_type = T;

    gpu_gemm_state(std::uint32_t N):
        N(N),
        a(malloc_device<value_type>(N*N)),
        b(malloc_device<value_type>(N*N)),
        c(malloc_device<value_type>(N*N)),
        beta(1./(N*N))
    {
        gpu_rand(a, N*N);
        gpu_rand(b, N*N);
        gpu_rand(c, N*N);
    }

    void run() {
        gpu_gemm(a, b, c, N, N, N, alpha, beta);
    }

    void synchronize() {
        cudaDeviceSynchronize();
    }

    std::string report(std::vector<double> times) {
        return flop_report_gemm(N, std::move(times));
    }

    ~gpu_gemm_state() {
        cudaFree(a);
        cudaFree(b);
        cudaFree(c);
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
struct gpu_stream_state: public benchmark {
    using value_type = T;

    gpu_stream_state(std::uint32_t N):
        N(N),
        a(malloc_device<value_type>(N)),
        b(malloc_device<value_type>(N)),
        c(malloc_device<value_type>(N))
    {
        gpu_rand(a, N);
        gpu_rand(b, N);
        gpu_rand(c, N);
    }

    void run() {
        gpu_stream_triad(a, b, c, alpha, N);
    }

    void synchronize() {
        cudaDeviceSynchronize();
    }

    std::string report(std::vector<double> times) {
        return bandwidth_report_stream(N, std::move(times));
    }

    ~gpu_stream_state() {
        cudaFree(a);
        cudaFree(b);
        cudaFree(c);
    }

private:
    const std::uint32_t N;
    const value_type alpha = 0.99;

    value_type* a;
    value_type* b;
    value_type* c;
};

std::unique_ptr<benchmark> get_gpu_benchmark(std::uint32_t N, benchmark_kind kind) {
    switch (kind) {
        case benchmark_kind::gemm:
            return std::make_unique<gpu_gemm_state<value_type>>(N);
        case benchmark_kind::stream:
            return std::make_unique<gpu_stream_state<value_type>>(N);
        default:
            return std::make_unique<null_benchmark>(N);
    }
}

