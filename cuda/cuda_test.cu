#include "kernels/cuda_check.h"
#include <iostream>
#include <cuda_runtime.h>

void print_cuda_info()
{
    int device_count = 0;

    cudaError_t err =
        cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess)
    {
        std::cerr
            << "cudaGetDeviceCount failed: "
            << cudaGetErrorString(err)
            << '\n';

        return;
    }

    std::cout
        << "CUDA device count: "
        << device_count
        << '\n';

    if (device_count > 0)
    {
        cudaDeviceProp prop{};

        cudaGetDeviceProperties(
            &prop,
            0);

        std::cout
            << "GPU: "
            << prop.name
            << '\n';
    }
}