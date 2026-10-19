#include "minillm/ops/rope.h"

#include <cuda_runtime.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace minillm
{

    namespace
    {

        constexpr int BLOCK_SIZE = 256;

        __global__ void rope_kernel(
            const float *input,
            float *output,
            std::size_t total_pairs,
            std::size_t heads,
            std::size_t head_dim,
            std::size_t start_pos,
            float theta_base)
        {
            const std::size_t idx =
                static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

            if (idx >= total_pairs)
            {
                return;
            }
            // [token, head, pair] -> [token, head, 2 * pair], [token, head, 2 * pair + 1]
            const std::size_t pairs_per_head = head_dim / 2;

            // 找到当前 Pair 所属的 Token 和 Head
            const std::size_t token_head =
                idx / pairs_per_head;
            // 找到当前 Pair 在 Head 中的索引
            const std::size_t pair =
                idx % pairs_per_head;

            const std::size_t token =
                token_head / heads;

            const std::size_t pos =
                start_pos + token;
            // Pair -> [2 * pair, 2 * pair + 1]，所以 base 是当前 Pair 的第一个元素的位置
            // 当前 Pair 的第一个元素的位置
            const std::size_t base =
                token_head * head_dim + 2 * pair;

            // θ = pos / theta_base^(2k/d)
            const float exponent =
                -2.0f * static_cast<float>(pair) / static_cast<float>(head_dim);

            const float angle =
                static_cast<float>(pos) *
                powf(theta_base, exponent);

            float s, c;
            sincosf(angle, &s, &c);

            const float x0 = input[base];
            const float x1 = input[base + 1];

            output[base] =
                x0 * c - x1 * s;

            output[base + 1] =
                x0 * s + x1 * c;
        }

    } // namespace

    void rope_cuda_async(
        const Tensor &input,
        Tensor &output,
        std::size_t start_pos,
        float theta_base,
        cudaStream_t stream)
    {
        if (input.device() != DeviceType::CUDA ||
            output.device() != DeviceType::CUDA)
        {
            throw std::invalid_argument(
                "RoPE requires CUDA tensors");
        }

        if (input.dtype() != DataType::FLOAT32 ||
            output.dtype() != DataType::FLOAT32 ||
            input.ndim() != 3 ||
            output.shape() != input.shape())
        {
            throw std::invalid_argument(
                "Invalid RoPE tensors");
        }

        const std::size_t seq_len = input.shape()[0];
        const std::size_t heads = input.shape()[1];
        const std::size_t head_dim = input.shape()[2];

        if (head_dim == 0 || head_dim % 2 != 0 ||
            !std::isfinite(theta_base) || theta_base <= 1.0f)
        {
            throw std::invalid_argument(
                "Invalid RoPE parameters");
        }

        if (seq_len > 0 &&
            start_pos >
                std::numeric_limits<std::size_t>::max() - (seq_len - 1))
        {
            throw std::overflow_error(
                "RoPE position overflow");
        }

        const std::size_t total_pairs =
            input.numel() / 2;

        if (total_pairs == 0)
        {
            return;
        }

        const std::size_t blocks =
            (total_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE;

        if (blocks >
            std::numeric_limits<unsigned int>::max())
        {
            throw std::length_error(
                "RoPE grid too large");
        }

        rope_kernel<<<
            static_cast<unsigned int>(blocks),
            BLOCK_SIZE,
            0,
            stream>>>(
            input.ptr<float>(),
            output.ptr<float>(),
            total_pairs,
            heads,
            head_dim,
            start_pos,
            theta_base);

        const cudaError_t err = cudaGetLastError();

        if (err != cudaSuccess)
        {
            throw std::runtime_error(
                std::string("RoPE launch error: ") + cudaGetErrorString(err));
        }
    }

    void rope_cuda(
        const Tensor &input,
        Tensor &output,
        std::size_t start_pos,
        float theta_base)
    {
        rope_cuda_async(
            input, output,
            start_pos,
            theta_base,
            nullptr);

        const cudaError_t err =
            cudaStreamSynchronize(nullptr);

        if (err != cudaSuccess)
        {
            throw std::runtime_error(
                std::string("RoPE execution error: ") + cudaGetErrorString(err));
        }
    }

} // namespace minillm