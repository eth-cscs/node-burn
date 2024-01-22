#ifdef USE_CUDA
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#endif
#include <type_traits>

#include "numeric.h"
#include "util.h"

#ifdef USE_CUDA
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

template void gpu_gemm(float* a, float* b, float* c,
          int m, int n, int k,
          float alpha, float beta);

template void gpu_gemm(double* a, double* b, double* c,
          int m, int n, int k,
          double alpha, double beta);
#endif

extern "C" void dgemm_(char*, char*, int*, int*,int*, double*, double*, int*, double*, int*, double*, double*, int*);
extern "C" void sgemm_(char*, char*, int*, int*,int*, float*, float*, int*, float*, int*, float*, float*, int*);

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

template void cpu_gemm(double* a, double*b, double*c,
                       int m, int n, int k,
                       double alpha, double beta);

template void cpu_gemm(float* a, float*b, float*c,
                       int m, int n, int k,
                       float alpha, float beta);

#ifdef USE_CUDA
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

template void gpu_rand(float*, uint64_t);
template void gpu_rand(double*, uint64_t);
#endif

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

template void cpu_rand(float*, uint64_t);
template void cpu_rand(double*, uint64_t);
