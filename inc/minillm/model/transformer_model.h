
#pragma once

#include <cstddef>
#include <memory>
#include <vector>
#include <cuda_runtime_api.h>

#include "minillm/model/model_config.h"
#include "minillm/model/embedding.h"
#include "minillm/layers/transformer_block.h"

namespace minillm
{

    class DecodeSession
    {
    public:
        explicit DecodeSession(const ModelConfig &config);

        CudaKVCache &cache(std::size_t layer)
        {
            return *caches_.at(layer);
        }

        const CudaKVCache &cache(std::size_t layer) const
        {
            return *caches_.at(layer);
        }

        std::size_t num_layers() const noexcept
        {
            return caches_.size();
        }

        void rewind_to(std::size_t position)
        {
            for (auto &cache : caches_)
                cache->rewind_to(position);
        }

        bool valid() const noexcept { return valid_; }

        void invalidate() noexcept { valid_ = false; }

    private:
        std::vector<std::unique_ptr<CudaKVCache>> caches_;
        bool valid_ = true;
    };

    class ModelWorkspace
    {
    public:
        explicit ModelWorkspace(const ModelConfig &config);

        Tensor hidden_a;     // [H]
        Tensor hidden_b;     // [H]
        Tensor final_hidden; // [H]
        Tensor logits;       // [V]

        std::vector<
            std::unique_ptr<TransformerBlockWorkspace>>
            layers;
    };

    class TransformerModel
    {
    public:
        explicit TransformerModel(ModelConfig config);

        Embedding &embedding() noexcept
        {
            return embedding_;
        }

        TransformerBlock &block(std::size_t i)
        {
            return *blocks_.at(i);
        }

        RMSNormLayer &final_norm() noexcept
        {
            return final_norm_;
        }

        // 仅适用于不共享 Embedding/LM Head 权重的模型
        void load_lm_head_weight(const Tensor &source);

        void forward_decode_async(
            std::size_t token_id,
            std::size_t position,
            DecodeSession &session,
            ModelWorkspace &workspace,
            cudaStream_t stream) const;

        void forward_decode(
            std::size_t token_id,
            std::size_t position,
            DecodeSession &session,
            ModelWorkspace &workspace) const;

    private:
        ModelConfig config_;

        Embedding embedding_;

        std::vector<
            std::unique_ptr<TransformerBlock>>
            blocks_;

        RMSNormLayer final_norm_;

        std::unique_ptr<Tensor> untied_lm_head_;
        bool lm_head_loaded_ = false;
    };

} // namespace minillm
