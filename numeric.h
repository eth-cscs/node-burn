#pragma once

#include <cstdint>

void gpu_gemm(double* a, double*b, double*c,
          int m, int n, int k,
          double alpha, double beta);

void cpu_gemm(double* a, double*b, double*c,
          int m, int n, int k,
          double alpha, double beta);

void gpu_rand(double* x, std::uint64_t n);
void cpu_rand(double* x, std::uint64_t n);

