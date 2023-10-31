#pragma once

#include <cstdint>

template<class T>
extern void gpu_gemm(T* a, T* b, T* c,
          int m, int n, int k,
          T alpha, T beta);

template<class T>
extern void cpu_gemm(T* a, T* b, T* c,
          int m, int n, int k,
          T alpha, T beta);

template<class T>
extern void gpu_rand(T* x, std::uint64_t n);

template<class T>
extern void cpu_rand(T* x, std::uint64_t n);

template<class T>
void cpu_stream_triad(T* __restrict__ a, T* __restrict__ b, T* __restrict__ c, T scale, std::uint64_t N)
{
#pragma omp parallel for schedule(static)
    for (std::uint64_t i = 0; i < N; ++i)
    {
        a[i] = b[i] + c[i] * scale;
    }
}
