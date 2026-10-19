#include "minillm/ops/softmax.h"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <limits>
#include <stdexcept>
#include <string>

namespace minillm
{

    namespace
    {

        constexpr int BLOCK_SIZE = 256;
        constexpr int WARP_SIZE = 32;
        constexpr int NUMS_WARPS = BLOCK_SIZE / WARP_SIZE;

        __device__ __forceinline__ float warp_reduce_max(float value)
        {
            for (int offset = WARP_SIZE / 2;
                 offset > 0;
                 offset /= 2)
            {
                value = fmaxf(
                    value,
                    __shfl_down_sync(0xffffffffu, value, offset));
            }

            return value;
        }

        __device__ __forceinline__ float warp_reduce_sum(float value)
        {
            for (int offset = WARP_SIZE / 2;
                 offset > 0;
                 offset /= 2)
            {
                value += __shfl_down_sync(
                    0xffffffffu,
                    value,
                    offset);
            }

            return value;
        }

        __global__ void softmax_kernel(
            const float *input,
            float *output,
            std::size_t hidden)
        {
            const int tid = threadIdx.x;

            const std::size_t row = blockIdx.x;
            const std::size_t base = row * hidden;

            __shared__ float partial[BLOCK_SIZE];
            __shared__ float row_max;
            __shared__ float row_inv_sum;

            // -------------------------
            // 第一阶段：求整行最大值
            // -------------------------

            float local_max = -CUDART_INF_F;

            for (std::size_t col = tid;
                 col < hidden;
                 col += blockDim.x)
            {
                local_max = fmaxf(
                    local_max,
                    input[base + col]);
            }

            partial[tid] = local_max;
            __syncthreads();

            // Max Reduction
            for (int stride = BLOCK_SIZE / 2;
                 stride > 0;
                 stride /= 2)
            {
                if (tid < stride)
                {
                    partial[tid] = fmaxf(
                        partial[tid],
                        partial[tid + stride]);
                }

                __syncthreads();
            }

            if (tid == 0)
            {
                row_max = partial[0];
            }

            __syncthreads();

            // -------------------------
            // 第二阶段：求 Exp Sum
            // -------------------------

            float local_sum = 0.0f;

            for (std::size_t col = tid;
                 col < hidden;
                 col += blockDim.x)
            {
                local_sum += expf(
                    input[base + col] - row_max);
            }

            // 复用 Shared Memory
            partial[tid] = local_sum;

            __syncthreads();

            // Sum Reduction
            for (int stride = BLOCK_SIZE / 2;
                 stride > 0;
                 stride /= 2)
            {
                if (tid < stride)
                {
                    partial[tid] +=
                        partial[tid + stride];
                }

                __syncthreads();
            }

            if (tid == 0)
            {
                row_inv_sum = 1.0f / partial[0];
            }

            __syncthreads();

            // -------------------------
            // 第三阶段：计算输出
            // -------------------------

            for (std::size_t col = tid;
                 col < hidden;
                 col += blockDim.x)
            {
                output[base + col] =
                    expf(input[base + col] - row_max) * row_inv_sum;
            }
        }

        __global__ void softmax_warp_kernel(
            const float *input,
            float *output,
            std::size_t rows,
            std::size_t hidden)
        {
            const int lane = threadIdx.x % WARP_SIZE;
            const int warp_id = threadIdx.x / WARP_SIZE;

            const std::size_t row =
                static_cast<std::size_t>(blockIdx.x) * NUMS_WARPS + warp_id;

            if (row >= rows)
            {
                return;
            }

            const std::size_t base = row * hidden;
            // 第一部 求每一行的最大值
            //  每个 lane 处理当前行的一部分元素，然后在 warp 内求最大值。
            float local_max = -CUDART_INF_F;

            for (std::size_t col = lane;
                 col < hidden;
                 col += WARP_SIZE)
            {
                local_max = fmaxf(
                    local_max,
                    input[base + col]);
            }

            local_max = warp_reduce_max(local_max);
            const float row_max = __shfl_sync(
                0xffffffffu,
                local_max,
                0);
            // 第二步求每一行的 exp 总和
            //  使用稳定 softmax：先减去行最大值，再计算 exp 总和。
            float local_sum = 0.0f;

            for (std::size_t col = lane;
                 col < hidden;
                 col += WARP_SIZE)
            {
                local_sum += expf(input[base + col] - row_max);
            }

            local_sum = warp_reduce_sum(local_sum);
            const float row_sum = __shfl_sync(
                0xffffffffu,
                local_sum,
                0);
            const float row_inv_sum = 1.0f / row_sum;
            // 第三步计算输出
            for (std::size_t col = lane;
                 col < hidden;
                 col += WARP_SIZE)
            {
                output[base + col] =
                    expf(input[base + col] - row_max) * row_inv_sum;
            }
        }

    }
    void softmax_cuda_async(
        const Tensor &input,
        Tensor &output,
        cudaStream_t stream,
        SoftmaxKernel kernel)
    {
        if (input.device() != DeviceType::CUDA ||
            output.device() != DeviceType::CUDA)
        {
            throw std::invalid_argument(
                "Softmax requires CUDA tensors");
        }

        if (input.dtype() != DataType::FLOAT32 ||
            output.dtype() != DataType::FLOAT32)
        {
            throw std::invalid_argument(
                "Softmax requires FLOAT32");
        }

        if (input.ndim() != 2 ||
            output.shape() != input.shape())
        {
            throw std::invalid_argument(
                "Softmax shape mismatch");
        }

        const std::size_t rows = input.shape()[0];
        const std::size_t hidden = input.shape()[1];

        if (hidden == 0)
        {
            throw std::invalid_argument(
                "Softmax hidden size must be positive");
        }

        if (rows == 0)
        {
            return;
        }

        const std::size_t block_count =
            kernel == SoftmaxKernel::Warp
                ? (rows + NUMS_WARPS - 1) / NUMS_WARPS
                : rows;

        if (block_count > static_cast<std::size_t>(
                              std::numeric_limits<unsigned int>::max()))
        {
            throw std::length_error(
                "Softmax grid too large");
        }

        if (kernel == SoftmaxKernel::Block)
        {
            softmax_kernel<<<
                static_cast<unsigned int>(block_count),
                BLOCK_SIZE,
                0,
                stream>>>(
                input.ptr<float>(),
                output.ptr<float>(),
                hidden);
        }
        else if (kernel == SoftmaxKernel::Warp)
        {
            softmax_warp_kernel<<<
                static_cast<unsigned int>(block_count),
                BLOCK_SIZE,
                0,
                stream>>>(
                input.ptr<float>(),
                output.ptr<float>(),
                rows,
                hidden);
        }
        else
        {
            throw std::invalid_argument(
                "Unsupported Softmax kernel");
        }

        const cudaError_t err = cudaGetLastError();

        if (err != cudaSuccess)
        {
            throw std::runtime_error(
                std::string("Softmax launch failed: ") + cudaGetErrorString(err));
        }
    }

    void softmax_cuda(
        const Tensor &input,
        Tensor &output,
        SoftmaxKernel kernel)
    {
        softmax_cuda_async(
            input,
            output,
            nullptr,
            kernel);

        const cudaError_t err =
            cudaStreamSynchronize(nullptr);

        if (err != cudaSuccess)
        {
            throw std::runtime_error(
                std::string("Softmax execution failed: ") + cudaGetErrorString(err));
        }
    }

} // namespace minillm
