#include "minillm/layers/self_attention.h"
#include "minillm/ops/rope.h"

#include <cuda_runtime_api.h>

#include <limits>
#include <stdexcept>
#include <vector>

namespace minillm
{

    namespace
    {

        std::size_t positive(std::size_t n)
        {
            if (n == 0)
            {
                throw std::invalid_argument(
                    "Dimension must be positive");
            }

            return n;
        }

        std::size_t checked_mul(
            std::size_t a,
            std::size_t b)
        {
            positive(a);
            positive(b);

            if (a >
                std::numeric_limits<std::size_t>::max() / b)
            {
                throw std::overflow_error(
                    "Shape multiplication overflow");
            }

            return a * b;
        }

        std::size_t checked_kv_heads(
            std::size_t num_q_heads,
            std::size_t num_kv_heads)
        {
            positive(num_q_heads);
            positive(num_kv_heads);

            if (num_q_heads % num_kv_heads != 0)
            {
                throw std::invalid_argument(
                    "Q heads must be divisible by KV heads");
            }

            return num_kv_heads;
        }

        std::size_t checked_head_dim(std::size_t d)
        {
            positive(d);

            if (d % 2 != 0)
            {
                throw std::invalid_argument(
                    "RoPE requires even head_dim");
            }

            return d;
        }

        void require_flat_cuda(
            const Tensor &tensor,
            std::size_t elements)
        {
            if (tensor.shape() !=
                    std::vector<std::size_t>{elements} ||
                tensor.device() != DeviceType::CUDA ||
                tensor.dtype() != DataType::FLOAT32 ||
                !tensor.is_contiguous())
            {
                throw std::invalid_argument(
                    "Expected flat contiguous CUDA FP32 Tensor");
            }
        }

        void require_workspace_tensor(
            const Tensor &tensor,
            std::size_t elements)
        {
            if (tensor.numel() != elements ||
                tensor.device() != DeviceType::CUDA ||
                tensor.dtype() != DataType::FLOAT32 ||
                !tensor.is_contiguous())
            {
                throw std::invalid_argument(
                    "Invalid SelfAttention workspace Tensor");
            }
        }

    } // namespace

    SelfAttentionWorkspace::SelfAttentionWorkspace(
        std::size_t num_q_heads,
        std::size_t num_kv_heads,
        std::size_t head_dim,
        std::size_t capacity)
        : q(
              {checked_mul(num_q_heads, head_dim)},
              DataType::FLOAT32,
              DeviceType::CUDA),
          k(
              {checked_mul(num_kv_heads, head_dim)},
              DataType::FLOAT32,
              DeviceType::CUDA),
          v(
              {checked_mul(num_kv_heads, head_dim)},
              DataType::FLOAT32,
              DeviceType::CUDA),
          q_rope(
              {1, positive(num_q_heads), positive(head_dim)},
              DataType::FLOAT32,
              DeviceType::CUDA),
          k_rope(
              {1, positive(num_kv_heads), positive(head_dim)},
              DataType::FLOAT32,
              DeviceType::CUDA),
          context(
              {num_q_heads, head_dim},
              DataType::FLOAT32,
              DeviceType::CUDA),
          attention(
              num_q_heads,
              positive(capacity))
    {
    }

    SelfAttention::SelfAttention(
        std::size_t hidden_size,
        std::size_t num_q_heads,
        std::size_t num_kv_heads,
        std::size_t head_dim)
        : hidden_size_(positive(hidden_size)),
          num_q_heads_(positive(num_q_heads)),
          num_kv_heads_(
              checked_kv_heads(num_q_heads_, num_kv_heads)),
          head_dim_(checked_head_dim(head_dim)),

          q_proj_(
              hidden_size_,
              checked_mul(num_q_heads_, head_dim_)),

          k_proj_(
              hidden_size_,
              checked_mul(num_kv_heads_, head_dim_)),

          v_proj_(
              hidden_size_,
              checked_mul(num_kv_heads_, head_dim_)),

          o_proj_(
              checked_mul(num_q_heads_, head_dim_),
              hidden_size_)
    {
    }

    void SelfAttention::forward_async(
        const Tensor &input,
        Tensor &output,
        CudaKVCache &cache,
        SelfAttentionWorkspace &workspace,
        std::size_t position,
        cudaStream_t stream) const
    {
        const std::size_t QD =
            checked_mul(num_q_heads_, head_dim_);

        const std::size_t KVD =
            checked_mul(num_kv_heads_, head_dim_);

        // =================================================
        // 1. 检查输入输出
        // =================================================
        require_flat_cuda(input, hidden_size_);
        require_flat_cuda(output, hidden_size_);

        // =================================================
        // 2. 检查 Workspace
        // =================================================
        require_workspace_tensor(workspace.q, QD);
        require_workspace_tensor(workspace.k, KVD);
        require_workspace_tensor(workspace.v, KVD);

        require_workspace_tensor(workspace.q_rope, QD);
        require_workspace_tensor(workspace.k_rope, KVD);
        require_workspace_tensor(workspace.context, QD);

        const std::vector<std::size_t> score_shape = {
            num_q_heads_, cache.capacity()};

        if (workspace.attention.scores.shape() != score_shape ||
            workspace.attention.weights.shape() != score_shape ||
            workspace.attention.scores.device() != DeviceType::CUDA ||
            workspace.attention.weights.device() != DeviceType::CUDA ||
            workspace.attention.scores.dtype() != DataType::FLOAT32 ||
            workspace.attention.weights.dtype() != DataType::FLOAT32 ||
            !workspace.attention.scores.is_contiguous() ||
            !workspace.attention.weights.is_contiguous())
        {
            throw std::invalid_argument(
                "Attention workspace mismatch");
        }

        // =================================================
        // 3. 检查 KV Cache 和 Token Position
        // =================================================
        if (cache.num_kv_heads() != num_kv_heads_ ||
            cache.head_dim() != head_dim_ ||
            cache.size() != position ||
            position >= cache.capacity())
        {
            throw std::invalid_argument(
                "KV Cache configuration or position mismatch");
        }

        // =================================================
        // 4. Q / K / V Linear Projection
        // =================================================

        // Linear 只接受一维输入输出。
        // 每次 Forward 都恢复为一维。
        workspace.q.reshape_({QD});
        workspace.k.reshape_({KVD});
        workspace.v.reshape_({KVD});

        // input [H] → q [Hq*D]
        q_proj_.forward_async(
            input, workspace.q, stream);

        // input [H] → k [Hkv*D]
        k_proj_.forward_async(
            input, workspace.k, stream);

        // input [H] → v [Hkv*D]
        v_proj_.forward_async(
            input, workspace.v, stream);

        // =================================================
        // 5. RoPE：转换成三维 Tensor
        // =================================================

        // q [Hq*D] → [1,Hq,D]
        workspace.q.reshape_({1, num_q_heads_, head_dim_});

        // k [Hkv*D] → [1,Hkv,D]
        workspace.k.reshape_({1, num_kv_heads_, head_dim_});

        // RoPE 的输出也使用三维 Shape
        workspace.q_rope.reshape_({1, num_q_heads_, head_dim_});

        workspace.k_rope.reshape_({1, num_kv_heads_, head_dim_});

        // 只旋转 Q 和 K
        rope_cuda_async(
            workspace.q,
            workspace.q_rope,
            position,
            10000.0f,
            stream);

        rope_cuda_async(
            workspace.k,
            workspace.k_rope,
            position,
            10000.0f,
            stream);

        // =================================================
        // 6. 将 K/V 追加到 KV Cache
        // =================================================

        // RoPE 输出：[1,Hkv,D] → [Hkv,D]
        workspace.k_rope.reshape_({num_kv_heads_, head_dim_});

        // V 不经过 RoPE：[Hkv*D] → [Hkv,D]
        workspace.v.reshape_({num_kv_heads_, head_dim_});

        cache.append_async(
            workspace.k_rope,
            workspace.v,
            position,
            stream);

        // =================================================
        // 7. CUDA Attention Decode
        // =================================================

        // Q：[1,Hq,D] → [Hq,D]
        workspace.q_rope.reshape_({num_q_heads_, head_dim_});

        // Context 要符合 Attention Output Shape
        workspace.context.reshape_({num_q_heads_, head_dim_});

        attention_decode_cuda_async(
            workspace.q_rope,
            cache,
            workspace.context,
            workspace.attention,
            stream);

        // =================================================
        // 8. Output Linear Projection
        // =================================================

        // Context：[Hq,D] → [Hq*D]
        workspace.context.reshape_({QD});

        // 输入 [Hq*D] → 输出 [hidden_size]
        o_proj_.forward_async(
            workspace.context,
            output,
            stream);
    }

    void SelfAttention::forward(
        const Tensor &input,
        Tensor &output,
        CudaKVCache &cache,
        SelfAttentionWorkspace &workspace,
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
