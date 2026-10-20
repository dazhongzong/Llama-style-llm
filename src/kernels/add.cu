#include "minillm/ops/add.h"

#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>

namespace minillm
{

    __global__ void add_kernel(
        const float *a,
        const float *b,
        float *c,
        std::size_t n)
    {
        std::size_t idx =
            static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

        if (idx < n)
        {
            c[idx] = a[idx] + b[idx];
        }
    }

    void add_cuda(
        const Tensor &a,
        const Tensor &b,
        Tensor &out)
    {
        add_cuda_async(a, b, out, nullptr);

        cudaError_t err = cudaDeviceSynchronize();

        if (err != cudaSuccess)
        {
            throw std::runtime_error(
                std::string("Kernel execution failed: ") + cudaGetErrorString(err));
        }
    }

    void add_cuda_async(
        const Tensor &a,
        const Tensor &b,
        Tensor &out,
        cudaStream_t stream)
    {
        if (a.device() != DeviceType::CUDA ||
            b.device() != DeviceType::CUDA ||
            out.device() != DeviceType::CUDA)
        {
            throw std::invalid_argument(
                "add_cuda_async requires CUDA tensors");
        }

        if (a.dtype() != DataType::FLOAT32 ||
            b.dtype() != DataType::FLOAT32 ||
            out.dtype() != DataType::FLOAT32)
        {
            throw std::invalid_argument(
                "add_cuda_async requires Float32 tensors");
        }

        if (a.shape() != b.shape() ||
            a.shape() != out.shape())
        {
            throw std::invalid_argument(
                "add_cuda_async shape mismatch");
        }

        const std::size_t n = a.numel();

        if (n == 0)
        {
            return;
        }

        constexpr unsigned int threads = 256;

        const std::size_t blocks =
            (n + threads - 1) / threads;

        if (blocks > std::numeric_limits<unsigned int>::max())
        {
            throw std::length_error(
                "Too many CUDA blocks");
        }

        add_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
            a.ptr<float>(),
            b.ptr<float>(),
            out.ptr<float>(),
            n);

        cudaError_t err = cudaGetLastError();

        if (err != cudaSuccess)
        {
            throw std::runtime_error(
                std::string("Kernel launch failed: ") + cudaGetErrorString(err));
        }
    }
}