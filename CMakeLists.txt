cmake_minimum_required(VERSION 3.22)
project(node-burn)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# C++ settings
enable_language(CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# fmt library
#add_subdirectory(external/fmt EXCLUDE_FROM_ALL)
#find_package(fmt REQUIRED)

add_executable(burn main.cpp util.cpp numeric.cpp stream_gpu.cu timers.cpp experiment.cpp external/fmt/src/format.cc external/fmt/src/os.cc)

find_package(CUDAToolkit 11 REQUIRED)
find_package(Threads REQUIRED)
find_package(BLAS REQUIRED)

enable_language(CUDA)

target_include_directories(burn PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(burn PRIVATE ${CMAKE_SOURCE_DIR}/external/fmt/include/)
target_include_directories(burn PRIVATE ${CMAKE_SOURCE_DIR}/external)

target_link_libraries(burn PRIVATE CUDA::cudart CUDA::cublas CUDA::curand Threads::Threads BLAS::BLAS)

