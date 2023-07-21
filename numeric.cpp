#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>

#include "numeric.h"
#include "util.h"

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

void gpu_gemm(double* a, double*b, double*c,
          int m, int n, int k,
          double alpha, double beta)
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

extern "C" void dgemm_(char*, char*, int*, int*,int*, double*, double*, int*, double*, int*, double*, double*, int*);

void cpu_gemm(double* a, double*b, double*c,
          int m, int n, int k,
          double alpha, double beta)
{
    char trans = 'N';
    dgemm_(&trans, &trans, &m, &n, &k, &alpha, a, &m, b, &k, &beta, c, &m);
}

void gpu_rand(double* x, uint64_t n) {
    curandGenerator_t gen;

    check_status(
            curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    check_status(
            curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
    check_status(
            curandGenerateNormalDouble(gen, x, n, 0., 1.));
    check_status(
            curandDestroyGenerator(gen));
}

// TODO: this should take additional arguments for the matrix dimensions, and
// use the i,j values as 32 bit inputs to the hashing generator.
void cpu_rand(double* x, uint64_t n) {
    auto xorshiftstar = [](uint64_t x) -> uint64_t {
        x ^= x >> 12; // a
        x ^= x << 25; // b
        x ^= x >> 27; // c
        return x * 0x2545F4914F6CDD1D;
    };

    auto generate_random_double = [&xorshiftstar](uint32_t u1, uint32_t u2) -> double {
        uint64_t combined = ((uint64_t)u1 << 32) | u2;
        uint64_t hashed = xorshiftstar(combined);
        return (double)hashed / (double)UINT64_MAX;
    };

    for (std::size_t i=0; i<n; ++i) {
        x[i] = generate_random_double(0, n) - 0.5;
    }
}

