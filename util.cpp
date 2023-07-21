#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>

#include "util.h"

void device_synchronize() {
    cudaDeviceSynchronize();
}

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
