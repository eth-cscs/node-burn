#pragma once

#include <cstdint>

#include <cstdlib>
#include <algorithm>

#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>

void device_synchronize();
void check_status(cudaError_t status);
void check_status(curandStatus_t status);

// allocate space on GPU for n instances of type T
template <typename T>
T* malloc_device(size_t n) {
    void* p;
    auto status = cudaMalloc(&p, n*sizeof(T));
    check_status(status);
    return (T*)p;
}
#endif

template <typename T>
T* malloc_host(std::size_t N, T value=T()) {
    T* ptr = static_cast<T*>(malloc(N*sizeof(T)));
    std::fill(ptr, ptr+N, value);

    return ptr;
}
