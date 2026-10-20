#include "minillm/ops/swiglu.h"

#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>
#include <string>

namespace minillm
{

    namespace
    {

        constexpr unsigned int BLOCK_SIZE = 256;

        __global__ void swiglu_kernel(
            const float *gate,
            const float *up,
            float *output,
            std::size_t n)
        {
            const std::size_t idx =
                static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

            if (idx >= n)
            {
                return;
            }

            const float g = gate[idx];
            const float u = up[idx];

            const float silu =
                g / (1.0f + expf(-g));

            output[idx] = silu * u;
        }

    } // namespace

    void swiglu_cuda_async(
        const Tensor &gate,
        const Tensor &up,
        Tensor &output,
        cudaStream_t stream)
    {
        if (gate.device() != DeviceType::CUDA ||
            up.device() != DeviceType::CUDA ||
            output.device() != DeviceType::CUDA)
        {
            throw std::invalid_argument(
                "SwiGLU requires CUDA tensors");
        }

        if (gate.dtype() != DataType::FLOAT32 ||
            up.dtype() != DataType::FLOAT32 ||
            output.dtype() != DataType::FLOAT32)
        {
            throw std::invalid_argument(
                "SwiGLU requires FLOAT32 tensors");
        }

        if (gate.shape() != up.shape() ||
            gate.shape() != output.shape())
        {
            throw std::invalid_argument(
                "SwiGLU shape mismatch");
        }

        const std::size_t n = gate.numel();

        if (n == 0)
        {
            return;
        }

        const std::size_t blocks =
            (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

        if (blocks >
            std::numeric_limits<unsigned int>::max())
        {
            throw std::length_error(
                "SwiGLU grid too large");
        }

        swiglu_kernel<<<
            static_cast<unsigned int>(blocks),
            BLOCK_SIZE,
            0,
            stream>>>(
            gate.ptr<float>(),
            up.ptr<float>(),
            output.ptr<float>(),
            n);

        const cudaError_t err = cudaGetLastError();

        if (err != cudaSuccess)
        {
            throw std::runtime_error(
                std::string("SwiGLU launch error: ") +
                cudaGetErrorString(err));
        }
    }

    void swiglu_cuda(
        const Tensor &gate,
        const Tensor &up,
        Tensor &output)
    {
        swiglu_cuda_async(
            gate,
            up,
            output,
            nullptr);

        const cudaError_t err =
            cudaStreamSynchronize(nullptr);

        if (err != cudaSuccess)
        {
            throw std::runtime_error(
                std::string("SwiGLU execution error: ") +
                cudaGetErrorString(err));
        }
    }

} // namespace minillm