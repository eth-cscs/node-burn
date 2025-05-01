# NODE BURN

Because sometimes burning the GPU is not enough.

A simple tool for running compute or memory intensive workloads on both CPU and GPU, in order to understand
* the performance of the individual components
* the impact of the workloads on one another
* the maximum power consumption of a node

## Features

Node burn can run GEMM or STREAM workloads on the GPU only, CPU only, or both simultaneously.

```bash
# run GEMM with matrix dimension 5000*5000 on the GPU,
# and STREAM triad with length 500000 on the CPU, for 30 seconds.
./burn -ggemm,5000 -cstream,500000 -d30

# run GEMM on the CPU, nothing on the GPU, for 3 minutes.
./burn -cgemm,5000 -d180

# run GEMM on the GPU, nothing on the CPU, for 20 seconds.
./burn -ggemm,5000 -d20
```

Sometimes we want to run multiple instances of node burn in a parallel job, e.g. 4 instances on a node with 4 GPUs, or one instance on every GPU in a cabinet, to ~see if anything catches on fire~ understand system behavior under load.
Use the `--batch` option to produce less verbose output that can be easily parsed by a post-processing script.

```bash
# on a system with 4 GPUs per node, use all 16 GPUs on 4 nodes to
# run GEMM with matrix dimension 10000*10000 on the GPU for 30 seconds
srun -n16 -N4 ./burn --batch -ggemm,10000 -d30
nid001272:gpu    584 iterations, 38930.92 GFlops,     30.0 seconds,    2.400 Gbytes
nid001272:gpu    579 iterations, 38555.99 GFlops,     30.0 seconds,    2.400 Gbytes
nid001272:gpu    561 iterations, 37348.14 GFlops,     30.0 seconds,    2.400 Gbytes
nid001272:gpu    600 iterations, 39939.47 GFlops,     30.0 seconds,    2.400 Gbytes
nid001278:gpu    585 iterations, 38994.35 GFlops,     30.0 seconds,    2.400 Gbytes
nid001278:gpu    584 iterations, 38914.98 GFlops,     30.0 seconds,    2.400 Gbytes
nid001278:gpu    589 iterations, 39200.59 GFlops,     30.1 seconds,    2.400 Gbytes
nid001278:gpu    589 iterations, 39204.37 GFlops,     30.0 seconds,    2.400 Gbytes
nid001274:gpu    557 iterations, 37091.74 GFlops,     30.0 seconds,    2.400 Gbytes
nid001274:gpu    560 iterations, 37289.96 GFlops,     30.0 seconds,    2.400 Gbytes
nid001274:gpu    542 iterations, 36090.85 GFlops,     30.0 seconds,    2.400 Gbytes
nid001274:gpu    503 iterations, 33473.36 GFlops,     30.1 seconds,    2.400 Gbytes
nid001276:gpu    584 iterations, 38929.67 GFlops,     30.0 seconds,    2.400 Gbytes
nid001276:gpu    589 iterations, 39253.24 GFlops,     30.0 seconds,    2.400 Gbytes
nid001276:gpu    588 iterations, 39170.08 GFlops,     30.0 seconds,    2.400 Gbytes
nid001276:gpu    589 iterations, 39224.21 GFlops,     30.0 seconds,    2.400 Gbytes
```

### Power Measurement on HPE-Cray systems

If running on a HPE Cray-EX system with `pm_counters`, nodeburn can be configured to generate a report of power consumption on each node. Enable it at build time with the `NB_PMCOUNTERS` CMake option (see below).

`node-burn` will generate power reports from all of the energy counters that it can detect on each node - the values reported will vary according to the node architecture.

## Requirements

C++20 for the C++ code, C++17 for the CUDA code.

It has only been tested with GCC 11+ and CUDA 11.8+.
* not tested with Clang, Intel, NVC, Cray compilers. It should work if the compiler is recent.

## Compiling

Node burn uses CMake to configure the build. There is currently one option, `NB_GPU` which can be used to to disable CUDA targets.

```bash
# by default node burn will attempt to build for CUDA devices.
CC=gcc CXX=g++ cmake $src_path

# explicitly disable building for CUDA
CC=gcc CXX=g++ cmake $src_path -DNB_GPU=off
```

On HPE Cray-EX systems, power readings from `pm_counters` can be generated using the `NB_PMCOUNTERS` option.
```bash
# enable pm counters for average power consumption
CC=gcc CXX=g++ cmake $src_path -DNB_PMCOUNTERS=on
```
