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

