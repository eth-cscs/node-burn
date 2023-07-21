#pragma once

#include <cstdint>

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

template <typename T>
T* malloc_host(size_t N, T value=T()) {
    T* ptr = (T*)(malloc(N*sizeof(T)));
    std::fill(ptr, ptr+N, value);

    return ptr;
}
