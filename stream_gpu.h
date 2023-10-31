#pragma once

#include <cstdint>

template<class T>
extern void gpu_stream_triad(T* a, const T* b, const T* c, T scale, std::uint64_t n);
