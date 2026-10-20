#include "minillm/ops/gemv.h"

#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>
#include <string>

namespace minillm
{

    namespace
    {

        constexpr int BLOCK_SIZE = 256;
        constexpr int WARP_SIZE = 32;
        constexpr int WARPS_PER_BLOCK =
            BLOCK_SIZE / WARP_SIZE;

        __device__ __forceinline__ float warp_reduce_sum(float value)
        {
            for (int offset = 16; offset > 0; offset /= 2)
            {
                value += __shfl_down_sync(
                    0xffffffffu,
                    value,
                    offset);
            }

            return value;
        }

        __global__ void gemv_kernel(
            const float *weight,
            const float *input,
            float *output,
            std::size_t M,
            std::size_t K)
        {
            // 1. 计算 Warp ID、Lane ID
            const int warp_id = threadIdx.x / WARP_SIZE;
            const int lane = threadIdx.x % WARP_SIZE;

            // 2. 每个 Warp 负责一行
            const std::size_t row =
                static_cast<std::size_t>(blockIdx.x) * WARPS_PER_BLOCK + warp_id;

            // 整个 Warp 对 row >= M 的判断一致
            if (row >= M)
            {
                return;
            }

            // 3. 每个 Lane 计算局部点积
            float local_sum = 0.0f;

            for (std::size_t col = lane;
                 col < K;
                 col += WARP_SIZE)
            {
                local_sum +=
                    weight[row * K + col] * input[col];
            }

            // 4. Warp 内归约
            local_sum = warp_reduce_sum(local_sum);

            // 5. Lane 0 写出整行结果
            if (lane == 0)
            {
                output[row] = local_sum;
            }
        }

        __global__ void gemv_vec4_kernel(
            const float4 *weight,
            const float4 *input,
            float *output,
            std::size_t M,
            std::size_t K4)
        {
            const int warp_id = threadIdx.x / WARP_SIZE;
            const int lane = threadIdx.x % WARP_SIZE;

            constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;

            const std::size_t row =
                static_cast<std::size_t>(blockIdx.x) * WARPS_PER_BLOCK + warp_id;

            if (row >= M)
            {
                return;
            }

            // 使用多个寄存器累加器
            float sum0 = 0.0f;
            float sum1 = 0.0f;
            float sum2 = 0.0f;
            float sum3 = 0.0f;

            // K4 = K / 4
            for (std::size_t col = lane;
                 col < K4;
                 col += WARP_SIZE)
            {
                const float4 w =
                    weight[row * K4 + col];

                const float4 x =
                    input[col];

                sum0 += w.x * x.x;
                sum1 += w.y * x.y;
                sum2 += w.z * x.z;
                sum3 += w.w * x.w;
            }

            // 合并当前 Lane 的四个累加器
            float local_sum =
                (sum0 + sum1) + (sum2 + sum3);

            // Warp Reduction
            local_sum = warp_reduce_sum(local_sum);
            if (lane == 0)
            {
                output[row] = local_sum;
            }
        }

    } // namespace

    void gemv_cuda(
        const Tensor &weight,
        const Tensor &input,
        Tensor &output)
    {
        gemv_cuda_async(
            weight,
            input,
            output,
            GemvKernel::Scalar,
            nullptr);

        const cudaError_t err =
            cudaStreamSynchronize(nullptr);

        if (err != cudaSuccess)
        {
            throw std::runtime_error(
                cudaGetErrorString(err));
        }
    }

    void gemv_cuda_async(
        const Tensor &weight,
        const Tensor &input,
        Tensor &output,
        GemvKernel version,
        cudaStream_t stream)
    {
        if (weight.device() != DeviceType::CUDA ||
            input.device() != DeviceType::CUDA ||
            output.device() != DeviceType::CUDA)
        {
            throw std::invalid_argument(
                "GEMV requires CUDA tensors");
        }

        if (weight.dtype() != DataType::FLOAT32 ||
            input.dtype() != DataType::FLOAT32 ||
            output.dtype() != DataType::FLOAT32)
        {
            throw std::invalid_argument(
                "GEMV requires FLOAT32");
        }

        if (weight.ndim() != 2 ||
            input.ndim() != 1 ||
            output.ndim() != 1)
        {
            throw std::invalid_argument(
                "Invalid GEMV dimensions");
        }

        const std::size_t M = weight.shape()[0];
        const std::size_t K = weight.shape()[1];

        if (K == 0 ||
            input.shape()[0] != K ||
            output.shape()[0] != M)
        {
            throw std::invalid_argument(
                "GEMV shape mismatch");
        }

        if (M == 0)
        {
            return;
        }

        constexpr unsigned int THREADS = 256;
        constexpr unsigned int WARPS_PER_BLOCK = 8;

        const std::size_t blocks =
            M / WARPS_PER_BLOCK +
            (M % WARPS_PER_BLOCK != 0);

        if (blocks >
            std::numeric_limits<unsigned int>::max())
        {
            throw std::length_error("GEMV grid too large");
        }

        const auto grid = static_cast<unsigned int>(blocks);

        if (version == GemvKernel::Scalar)
        {
            gemv_kernel<<<grid, THREADS, 0, stream>>>(
                weight.ptr<float>(),
                input.ptr<float>(),
                output.ptr<float>(),
                M,
                K);
        }
        else if (version == GemvKernel::Vec4)
        {
            const auto aligned16 = [](const void *p)
            {
                return reinterpret_cast<std::uintptr_t>(p) % 16 == 0;
            };

            if (K % 4 != 0 ||
                !aligned16(weight.data()) ||
                !aligned16(input.data()))
            {
                throw std::invalid_argument(
                    "Vec4 requires 16-byte alignment and K % 4 == 0");
            }

            gemv_vec4_kernel<<<grid, THREADS, 0, stream>>>(
                reinterpret_cast<const float4 *>(
                    weight.ptr<float>()),
                reinterpret_cast<const float4 *>(
                    input.ptr<float>()),
                output.ptr<float>(),
                M,
                K / 4);
        }
        else
        {
            throw std::invalid_argument(
                "Unsupported GEMV kernel version");
        }

        const cudaError_t err = cudaGetLastError();

        if (err != cudaSuccess)
        {
            throw std::runtime_error(
                cudaGetErrorString(err));
        }

        // 注意：这里不调用 cudaDeviceSynchronize()
    }

} // namespace minillm