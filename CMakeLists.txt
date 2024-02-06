cmake_minimum_required(VERSION 3.22)
project(node-burn)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# C++ settings
enable_language(CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# By default try to build GPU.
# CPU only requires GPU to be explicitly disabled.
option(NB_GPU "GPU target: one of (none, cuda)." ON)

set(burn_sources
    main.cpp
    cpu_benchmark.cpp
    timers.cpp
    experiment.cpp
    external/fmt/src/format.cc
    external/fmt/src/os.cc
)

if(NB_GPU)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        set(CMAKE_CUDA_STANDARD 17)
        enable_language(CUDA)
        list(APPEND burn_sources
            gpu_benchmark.cpp
            stream_gpu.cu
        )
    endif()
endif()

add_executable(burn ${burn_sources})

find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)
find_package(BLAS REQUIRED)

target_include_directories(burn PRIVATE ${CMAKE_SOURCE_DIR}/external/fmt/include/)
target_include_directories(burn PRIVATE ${CMAKE_SOURCE_DIR}/external)
target_link_libraries(burn PRIVATE Threads::Threads OpenMP::OpenMP_CXX BLAS::BLAS)
if(NB_GPU)
    find_package(CUDAToolkit 11 REQUIRED)
    target_compile_definitions(burn PRIVATE NB_GPU)
    target_include_directories(burn PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(burn PRIVATE CUDA::cudart CUDA::cublas CUDA::curand)
endif()


