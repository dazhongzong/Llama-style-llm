#include "minillm/layers/attention_cuda.h"
#include "minillm/ops/softmax.h"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace minillm
{
    namespace
    {

        constexpr int BLOCK_SIZE = 256;
        constexpr int WARP_SIZE = 32;
        constexpr int WARPS_PER_BLOCK = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;

        __global__ void attention_scores_kernel(
            const float *q,
            const float *keys,
            float *scores,
            std::size_t context_len,
            std::size_t capacity,
            std::size_t num_kv_heads,
            std::size_t head_dim,
            std::size_t group_size,
            float scale)
        {
            // Grid.y 对应 Q Head
            const std::size_t q_head = blockIdx.y;

            const int warp_id =
                threadIdx.x / WARP_SIZE;

            const int lane =
                threadIdx.x % WARP_SIZE;

            // 一个 Warp 负责一个历史 Token
            const std::size_t token =
                static_cast<std::size_t>(blockIdx.x) * WARPS_PER_BLOCK + warp_id;

            if (token >= capacity)
            {
                return;
            }

            // 无效位置显式写 -inf
            if (token >= context_len)
            {
                if (lane == 0)
                {
                    scores[q_head * capacity + token] =
                        -CUDART_INF_F;
                }
                return;
            }

            // GQA 映射
            const std::size_t kv_head =
                q_head / group_size;

            // 当前 Query Head
            const float *q_ptr =
                q + q_head * head_dim;

            // 历史 Token 的对应 Key Head
            const float *k_ptr =
                keys +
                (token * num_kv_heads + kv_head) * head_dim;

            // 每个 Lane 计算局部点积
            float local_sum = 0.0f;

            for (std::size_t d = lane;
                 d < head_dim;
                 d += WARP_SIZE)
            {
                local_sum +=
                    q_ptr[d] * k_ptr[d];
            }

            // Warp Reduction
            for (int offset = 16;
                 offset > 0;
                 offset /= 2)
            {
                local_sum += __shfl_down_sync(
                    0xffffffffu,
                    local_sum,
                    offset);
            }

            // 只有 Lane 0 写 Score
            if (lane == 0)
            {
                scores[q_head * capacity + token] =
                    local_sum * scale;
            }
        }

        __global__ void attention_weighted_sum_kernel(
            const float *weights,
            const float *values,
            float *output,
            std::size_t context_len,
            std::size_t capacity,
            std::size_t num_kv_heads,
            std::size_t head_dim,
            std::size_t group_size)
        {
            const std::size_t q_head = blockIdx.y;

            const std::size_t d =
                static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

            if (d >= head_dim)
            {
                return;
            }

            const std::size_t kv_head =
                q_head / group_size;

            float sum = 0.0f;

            for (std::size_t t = 0;
                 t < context_len;
                 ++t)
            {
                const float p =
                    weights[q_head * capacity + t];

                const float v =
                    values[(t * num_kv_heads + kv_head) * head_dim + d];

                sum += p * v;
            }

            output[q_head * head_dim + d] = sum;
        }

    } // namespace
    AttentionWorkspace::AttentionWorkspace(
        std::size_t num_q_heads,
        std::size_t capacity)
        : scores(
              {num_q_heads, capacity},
              DataType::FLOAT32,
              DeviceType::CUDA),
          weights(
              {num_q_heads, capacity},
              DataType::FLOAT32,
              DeviceType::CUDA)
    {
    }

    void attention_decode_cuda_async(
        const Tensor &q,
        const CudaKVCache &cache,
        Tensor &output,
        AttentionWorkspace &workspace,
        cudaStream_t stream)
    {
        if (q.device() != DeviceType::CUDA ||
            output.device() != DeviceType::CUDA ||
            q.dtype() != DataType::FLOAT32 ||
            output.dtype() != DataType::FLOAT32 ||
            q.ndim() != 2 ||
            output.shape() != q.shape())
        {
            throw std::invalid_argument(
                "Invalid CUDA Attention tensors");
        }

        const std::size_t Hq = q.shape()[0];
        const std::size_t D = q.shape()[1];

        const std::size_t Hkv = cache.num_kv_heads();
        const std::size_t capacity = cache.capacity();
        const std::size_t T = cache.size();

        if (Hq == 0 ||
            Hkv == 0 ||
            Hq % Hkv != 0 ||
            D == 0 ||
            D != cache.head_dim() ||
            T == 0 ||
            T > capacity)
        {
            throw std::invalid_argument(
                "Invalid Attention configuration");
        }

        const std::vector<std::size_t> expected = {
            Hq, capacity};

        if (workspace.scores.shape() != expected ||
            workspace.weights.shape() != expected ||
            workspace.scores.device() != DeviceType::CUDA ||
            workspace.weights.device() != DeviceType::CUDA ||
            workspace.scores.dtype() != DataType::FLOAT32 ||
            workspace.weights.dtype() != DataType::FLOAT32)
        {
            throw std::invalid_argument(
                "Invalid Attention workspace");
        }

        // 本教学版使用二维 Grid，限制 Grid.y
        if (Hq > 65535)
        {
            throw std::length_error(
                "Too many Query Heads");
        }

        const std::size_t score_blocks =
            (capacity + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

        const std::size_t value_blocks =
            (D + BLOCK_SIZE - 1) / BLOCK_SIZE;

        if (score_blocks >
                std::numeric_limits<unsigned int>::max() ||
            value_blocks >
                std::numeric_limits<unsigned int>::max())
        {
            throw std::length_error(
                "Attention grid too large");
        }

        const std::size_t group_size = Hq / Hkv;

        const float scale =
            1.0f / std::sqrt(static_cast<float>(D));

        // 1. QK^T / sqrt(D)
        const dim3 score_grid(
            static_cast<unsigned int>(score_blocks),
            static_cast<unsigned int>(Hq));

        attention_scores_kernel<<<
            score_grid,
            BLOCK_SIZE,
            0,
            stream>>>(
            q.ptr<float>(),
            cache.keys_device_data(),
            workspace.scores.ptr<float>(),
            T,
            capacity,
            Hkv,
            D,
            group_size,
            scale);

        cudaError_t err = cudaGetLastError();

        if (err != cudaSuccess)
        {
            throw std::runtime_error(
                cudaGetErrorString(err));
        }

        // 2. Softmax
        softmax_cuda_async(
            workspace.scores,
            workspace.weights,
            stream);

        // 3. Weights × V
        const dim3 value_grid(
            static_cast<unsigned int>(value_blocks),
            static_cast<unsigned int>(Hq));

        attention_weighted_sum_kernel<<<
            value_grid,
            BLOCK_SIZE,
            0,
            stream>>>(
            workspace.weights.ptr<float>(),
            cache.values_device_data(),
            output.ptr<float>(),
            T,
            capacity,
            Hkv,
            D,
            group_size);

        err = cudaGetLastError();

        if (err != cudaSuccess)
        {
            throw std::runtime_error(
                cudaGetErrorString(err));
        }
    }

    void attention_decode_cuda(
        const Tensor &q,
        const CudaKVCache &cache,
        Tensor &output,
        AttentionWorkspace &workspace)
    {
        attention_decode_cuda_async(
            q, cache, output, workspace, nullptr);

        const cudaError_t err =
            cudaStreamSynchronize(nullptr);

        if (err != cudaSuccess)
        {
            throw std::runtime_error(
                cudaGetErrorString(err));
        }
    }
}