#include "minillm/model/transformer_model.h"

#include "minillm/ops/gemv.h"

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <utility>
#include <vector>

namespace minillm
{

    DecodeSession::DecodeSession(
        const ModelConfig &config)
    {
        config.validate();

        caches_.reserve(config.num_layers);

        for (std::size_t i = 0;
             i < config.num_layers; ++i)
        {
            caches_.push_back(
                std::make_unique<CudaKVCache>(
                    config.kv_capacity,
                    config.num_kv_heads,
                    config.head_dim));
        }
    }

    ModelWorkspace::ModelWorkspace(
        const ModelConfig &config)
        : hidden_a(
              {config.hidden_size},
              DataType::FLOAT32,
              DeviceType::CUDA),
          hidden_b(
              {config.hidden_size},
              DataType::FLOAT32,
              DeviceType::CUDA),
          final_hidden(
              {config.hidden_size},
              DataType::FLOAT32,
              DeviceType::CUDA),
          logits(
              {config.vocab_size},
              DataType::FLOAT32,
              DeviceType::CUDA)
    {
        config.validate();

        layers.reserve(config.num_layers);

        for (std::size_t i = 0;
             i < config.num_layers; ++i)
        {
            layers.push_back(
                std::make_unique<TransformerBlockWorkspace>(
                    config.hidden_size,
                    config.intermediate_size,
                    config.num_q_heads,
                    config.num_kv_heads,
                    config.head_dim,
                    config.kv_capacity));
        }
    }
    TransformerModel::TransformerModel(
        ModelConfig config)
        : config_(std::move(config)),
          embedding_(
              config_.vocab_size,
              config_.hidden_size),
          final_norm_(
              config_.hidden_size,
              config_.rms_norm_eps)
    {
        config_.validate();

        blocks_.reserve(config_.num_layers);

        for (std::size_t i = 0;
             i < config_.num_layers; ++i)
        {
            blocks_.push_back(
                std::make_unique<TransformerBlock>(
                    config_.hidden_size,
                    config_.intermediate_size,
                    config_.num_q_heads,
                    config_.num_kv_heads,
                    config_.head_dim,
                    config_.rms_norm_eps,
                    RoPEConfig{
                        config_.rope_theta,
                        config_.rope_layout}));
        }

        // 对于未共享权重的模型，额外申请 LM Head
        if (!config_.tie_word_embeddings)
        {
            untied_lm_head_ = std::make_unique<Tensor>(
                std::vector<std::size_t>{
                    config_.vocab_size,
                    config_.hidden_size},
                DataType::FLOAT32,
                DeviceType::CUDA);
        }
    }

    void TransformerModel::load_lm_head_weight(
        const Tensor &source)
    {
        if (!untied_lm_head_)
        {
            throw std::logic_error(
                "This model uses tied word embeddings");
        }

        if (source.shape() !=
                untied_lm_head_->shape() ||
            source.dtype() != DataType::FLOAT32 ||
            !source.is_contiguous())
        {
            throw std::invalid_argument(
                "Invalid LM Head weight");
        }

        untied_lm_head_->copy_from(source);
        lm_head_loaded_ = true;
    }
    void TransformerModel::forward_decode_async(
        std::size_t token_id,
        std::size_t position,
        DecodeSession &session,
        ModelWorkspace &workspace,
        cudaStream_t stream) const
    {
        // =====================================
        // 1. 先检查参数和 Session 状态
        // =====================================
        if (!session.valid())
        {
            throw std::runtime_error(
                "DecodeSession is invalid");
        }

        if (token_id >= config_.vocab_size ||
            position >= config_.kv_capacity ||
            position >= config_.max_position_embeddings ||
            session.num_layers() != config_.num_layers ||
            workspace.layers.size() != config_.num_layers)
        {
            throw std::invalid_argument(
                "Invalid decode input or workspace");
        }

        // 所有 Layer 必须位于相同 Token Position
        for (std::size_t i = 0;
             i < config_.num_layers; ++i)
        {
            if (session.cache(i).size() != position)
            {
                throw std::invalid_argument(
                    "Layer KV Cache positions are inconsistent");
            }
        }

        if (!embedding_.loaded())
        {
            throw std::runtime_error(
                "Embedding weight not loaded");
        }

        if (!config_.tie_word_embeddings &&
            !lm_head_loaded_)
        {
            throw std::runtime_error(
                "LM Head weight not loaded");
        }

        const std::vector<std::size_t> hidden_shape{
            config_.hidden_size};
        const std::vector<std::size_t> logits_shape{
            config_.vocab_size};

        auto valid_hidden = [&](const Tensor &t)
        {
            return t.shape() == hidden_shape &&
                   t.dtype() == DataType::FLOAT32 &&
                   t.device() == DeviceType::CUDA &&
                   t.is_contiguous();
        };

        if (!valid_hidden(workspace.hidden_a) ||
            !valid_hidden(workspace.hidden_b) ||
            !valid_hidden(workspace.final_hidden) ||
            workspace.logits.shape() != logits_shape ||
            workspace.logits.dtype() != DataType::FLOAT32 ||
            workspace.logits.device() != DeviceType::CUDA ||
            !workspace.logits.is_contiguous())
        {
            throw std::invalid_argument(
                "Invalid model workspace buffers");
        }

        try
        {
            // =====================================
            // 2. Embedding：Token ID → [H]
            // =====================================
            embedding_.forward_async(
                token_id,
                workspace.hidden_a,
                stream);

            // =====================================
            // 3. 双缓冲：循环执行所有 TransformerBlock
            // =====================================
            Tensor *current = &workspace.hidden_a;
            Tensor *next = &workspace.hidden_b;

            for (std::size_t i = 0;
                 i < config_.num_layers; ++i)
            {
                // current [H] → next [H]
                blocks_[i]->forward_async(
                    *current,
                    *next,
                    session.cache(i),
                    *workspace.layers[i],
                    position,
                    stream);

                // 只交换指针，不拷贝 GPU 数据
                std::swap(current, next);
            }

            // =====================================
            // 4. Final RMSNorm
            // =====================================
            final_norm_.forward_async(
                *current,
                workspace.final_hidden,
                stream);

            // =====================================
            // 5. LM Head：[H] → [V]
            // =====================================
            const Tensor &lm_weight =
                config_.tie_word_embeddings
                    ? embedding_.weight()
                    : *untied_lm_head_;

            gemv_cuda_async(
                lm_weight,              // [V,H]
                workspace.final_hidden, // [H]
                workspace.logits,       // [V]
                GemvKernel::Scalar,
                stream);
        }
        catch (...)
        {
            // 可能已有部分 Layer 提交了 GPU 任务。
            // 不允许把这种 Session 当作成功状态继续使用。
            session.invalidate();
            throw;
        }
    }

    void TransformerModel::forward_decode(
        std::size_t token_id,
        std::size_t position,
        DecodeSession &session,
        ModelWorkspace &workspace) const
    {
        forward_decode_async(
            token_id,
            position,
            session,
            workspace,
            nullptr);

        const cudaError_t err =
            cudaStreamSynchronize(nullptr);

        if (err != cudaSuccess)
        {
            session.invalidate();

            throw std::runtime_error(
                cudaGetErrorString(err));
        }
    }

} // namespace minillm
