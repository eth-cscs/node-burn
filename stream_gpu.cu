#include <cuda_runtime.h>

#include "stream_gpu.h"

template<class T>
__global__ void stream_triad(T* a, const T* b, const T* c, T scale, std::uint64_t n)
{
    std::uint64_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        a[i] = b[i] + c[i] * scale;
    }
}

template<class T>
void gpu_stream_triad(T* a, const T* b, const T* c, T scale, std::uint64_t n)
{
    unsigned numThreads = 256;
    unsigned numBlocks  = (n - 1) / numThreads + 1;
    stream_triad<<<numBlocks, numThreads>>>(a, b, c, scale, n);
}

template void gpu_stream_triad(float*, const float*, const float*, float, std::uint64_t);
template void gpu_stream_triad(double*, const double*, const double*, double, std::uint64_t);
