#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>

#include "minillm/layers/linear.h"
#include "minillm/layers/cuda_kv_cache.h"
#include "minillm/layers/attention_cuda.h"
#include "minillm/ops/rope_config.h"
namespace minillm
{

    class SelfAttentionWorkspace
    {
    public:
        SelfAttentionWorkspace(
            std::size_t num_q_heads,
            std::size_t num_kv_heads,
            std::size_t head_dim,
            std::size_t capacity);

        Tensor q; // 初始 [Hq*D]
        Tensor k; // 初始 [Hkv*D]
        Tensor v; // 初始 [Hkv*D]

        Tensor q_rope; // 初始 [1,Hq,D]
        Tensor k_rope; // 初始 [1,Hkv,D]

        Tensor context; // 初始 [Hq,D]

        AttentionWorkspace attention;
    };

    class SelfAttention
    {
    public:
        SelfAttention(
            std::size_t hidden_size,
            std::size_t num_q_heads,
            std::size_t num_kv_heads,
            std::size_t head_dim,
            RoPEConfig rope_config);

        Linear &q_proj() noexcept { return q_proj_; }
        Linear &k_proj() noexcept { return k_proj_; }
        Linear &v_proj() noexcept { return v_proj_; }
        Linear &o_proj() noexcept { return o_proj_; }

        void forward_async(
            const Tensor &input,
            Tensor &output,
            CudaKVCache &cache,
            SelfAttentionWorkspace &workspace,
            std::size_t position,
            cudaStream_t stream) const;

        void forward(
            const Tensor &input,
            Tensor &output,
            CudaKVCache &cache,
            SelfAttentionWorkspace &workspace,
            std::size_t position) const;

    private:
        std::size_t hidden_size_;
        std::size_t num_q_heads_;
        std::size_t num_kv_heads_;
        std::size_t head_dim_;

        Linear q_proj_;
        Linear k_proj_;
        Linear v_proj_;
        Linear o_proj_;

        RoPEConfig rope_config_;
    };

} // namespace minillm