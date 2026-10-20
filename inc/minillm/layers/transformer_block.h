#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>

#include "minillm/layers/rmsnorm_layer.h"
#include "minillm/layers/self_attention.h"
#include "minillm/layers/feed_forward.h"

namespace minillm
{

    class TransformerBlockWorkspace
    {
    public:
        TransformerBlockWorkspace(
            std::size_t hidden_size,
            std::size_t intermediate_size,
            std::size_t num_q_heads,
            std::size_t num_kv_heads,
            std::size_t head_dim,
            std::size_t capacity);

        Tensor norm1_out; // [H]
        Tensor attn_out;  // [H]
        Tensor residual;  // [H]
        Tensor norm2_out; // [H]
        Tensor ffn_out;   // [H]

        SelfAttentionWorkspace attention;
        FeedForwardWorkspace ffn;
    };

    class TransformerBlock
    {
    public:
        TransformerBlock(
            std::size_t hidden_size,
            std::size_t intermediate_size,
            std::size_t num_q_heads,
            std::size_t num_kv_heads,
            std::size_t head_dim,
            float rms_eps = 1e-5f,
            RoPEConfig rope_config = {});

        RMSNormLayer &input_norm() noexcept
        {
            return input_norm_;
        }

        SelfAttention &attention() noexcept
        {
            return attention_;
        }

        RMSNormLayer &post_attention_norm() noexcept
        {
            return post_attention_norm_;
        }

        FeedForward &feed_forward() noexcept
        {
            return feed_forward_;
        }

        void forward_async(
            const Tensor &input,
            Tensor &output,
            CudaKVCache &cache,
            TransformerBlockWorkspace &workspace,
            std::size_t position,
            cudaStream_t stream) const;

        void forward(
            const Tensor &input,
            Tensor &output,
            CudaKVCache &cache,
            TransformerBlockWorkspace &workspace,
            std::size_t position) const;

    private:
        std::size_t hidden_size_;
        std::size_t intermediate_size_;

        RMSNormLayer input_norm_;
        SelfAttention attention_;
        RMSNormLayer post_attention_norm_;
        FeedForward feed_forward_;
    };

} // namespace minillm