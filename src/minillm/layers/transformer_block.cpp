#include "minillm/layers/transformer_block.h"
#include "minillm/ops/add.h"

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <vector>

namespace minillm
{

    namespace
    {

        std::size_t positive(std::size_t n)
        {
            if (n == 0)
                throw std::invalid_argument(
                    "TransformerBlock dimension must be positive");

            return n;
        }

        void require_hidden_tensor(
            const Tensor &tensor,
            std::size_t hidden_size)
        {
            if (tensor.shape() !=
                    std::vector<std::size_t>{hidden_size} ||
                tensor.device() != DeviceType::CUDA ||
                tensor.dtype() != DataType::FLOAT32 ||
                !tensor.is_contiguous())
            {
                throw std::invalid_argument(
                    "Expected contiguous CUDA FP32 [hidden_size]");
            }
        }

    } // namespace
    TransformerBlockWorkspace::TransformerBlockWorkspace(
        std::size_t hidden_size,
        std::size_t intermediate_size,
        std::size_t num_q_heads,
        std::size_t num_kv_heads,
        std::size_t head_dim,
        std::size_t capacity)
        : norm1_out(
              {positive(hidden_size)},
              DataType::FLOAT32,
              DeviceType::CUDA),

          attn_out(
              {hidden_size},
              DataType::FLOAT32,
              DeviceType::CUDA),

          residual(
              {hidden_size},
              DataType::FLOAT32,
              DeviceType::CUDA),

          norm2_out(
              {hidden_size},
              DataType::FLOAT32,
              DeviceType::CUDA),

          ffn_out(
              {hidden_size},
              DataType::FLOAT32,
              DeviceType::CUDA),

          attention(
              num_q_heads,
              num_kv_heads,
              head_dim,
              capacity),

          ffn(positive(intermediate_size))
    {
    }

    TransformerBlock::TransformerBlock(
        std::size_t hidden_size,
        std::size_t intermediate_size,
        std::size_t num_q_heads,
        std::size_t num_kv_heads,
        std::size_t head_dim,
        float rms_eps,
        RoPEConfig rope_config)
        : hidden_size_(positive(hidden_size)),
          intermediate_size_(positive(intermediate_size)),

          input_norm_(hidden_size_, rms_eps),

          attention_(
              hidden_size_,
              num_q_heads,
              num_kv_heads,
              head_dim,
              rope_config),

          post_attention_norm_(hidden_size_, rms_eps),

          feed_forward_(
              hidden_size_,
              intermediate_size_)
    {
    }

    void TransformerBlock::forward_async(
        const Tensor &input,
        Tensor &output,
        CudaKVCache &cache,
        TransformerBlockWorkspace &workspace,
        std::size_t position,
        cudaStream_t stream) const
    {
        // =============================================
        // 1. 检查输入和输出
        // =============================================
        require_hidden_tensor(input, hidden_size_);
        require_hidden_tensor(output, hidden_size_);

        // =============================================
        // 2. 检查主干 Workspace
        // =============================================
        require_hidden_tensor(
            workspace.norm1_out, hidden_size_);

        require_hidden_tensor(
            workspace.attn_out, hidden_size_);

        require_hidden_tensor(
            workspace.residual, hidden_size_);

        require_hidden_tensor(
            workspace.norm2_out, hidden_size_);

        require_hidden_tensor(
            workspace.ffn_out, hidden_size_);

        // 当前是顺序 Decode，Position 必须连续
        if (position != cache.size() ||
            position >= cache.capacity())
        {
            throw std::invalid_argument(
                "Invalid TransformerBlock position");
        }

        // =============================================
        // 3. 第一层 RMSNorm
        // =============================================

        // input [H] -> norm1_out [H]
        input_norm_.forward_async(
            input,
            workspace.norm1_out,
            stream);

        // =============================================
        // 4. SelfAttention
        // =============================================

        // norm1_out [H] -> attn_out [H]
        // 内部完成 QKV、RoPE、KV Cache、Attention、Wo
        attention_.forward_async(
            workspace.norm1_out,
            workspace.attn_out,
            cache,
            workspace.attention,
            position,
            stream);

        // =============================================
        // 5. 第一次 Residual Add
        // =============================================

        // residual = input + attn_out
        add_cuda_async(
            input,
            workspace.attn_out,
            workspace.residual,
            stream);

        // =============================================
        // 6. 第二层 RMSNorm
        // =============================================

        // residual [H] -> norm2_out [H]
        post_attention_norm_.forward_async(
            workspace.residual,
            workspace.norm2_out,
            stream);

        // =============================================
        // 7. FeedForward
        // =============================================

        // norm2_out [H] -> ffn_out [H]
        feed_forward_.forward_async(
            workspace.norm2_out,
            workspace.ffn_out,
            workspace.ffn,
            stream);

        // =============================================
        // 8. 第二次 Residual Add
        // =============================================

        // output = residual + ffn_out
        add_cuda_async(
            workspace.residual,
            workspace.ffn_out,
            output,
            stream);
    }

    void TransformerBlock::forward(
        const Tensor &input,
        Tensor &output,
        CudaKVCache &cache,
        TransformerBlockWorkspace &workspace,
        std::size_t position) const
    {
        forward_async(
            input,
            output,
            cache,
            workspace,
            position,
            nullptr);

        const cudaError_t err =
            cudaStreamSynchronize(nullptr);

        if (err != cudaSuccess)
        {
            throw std::runtime_error(
                cudaGetErrorString(err));
        }
    }

} // namespace minillm