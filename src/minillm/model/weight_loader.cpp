
#include "minillm/model/weight_loader.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace minillm
{

    void load_model_weights(
        TransformerModel &model,
        const ModelConfig &config,
        const SafeTensorsReader &reader)
    {
        using Shape = std::vector<std::size_t>;

        const std::size_t H = config.hidden_size;
        const std::size_t I = config.intermediate_size;
        const std::size_t V = config.vocab_size;

        const std::size_t QD =
            config.num_q_heads * config.head_dim;

        const std::size_t KVD =
            config.num_kv_heads * config.head_dim;

        // 读取一个权重，并转换成临时 CPU Tensor。
        // 调用方负责把它加载进对应模型模块。
        auto with_weight = [&](
                               const std::string &name,
                               const Shape &expected_shape,
                               auto &&load)
        {
            HostWeightF32 data = reader.read_f32(name);

            if (data.shape != expected_shape)
            {
                throw std::runtime_error(
                    "Weight shape mismatch: " + name);
            }

            Tensor cpu_weight(
                expected_shape,
                DataType::FLOAT32,
                DeviceType::CPU);

            std::copy(
                data.values.begin(),
                data.values.end(),
                cpu_weight.ptr<float>());

            // 这里调用原来各 Layer 的 load_weight()
            load(cpu_weight);

            // 确保加载操作不再使用临时 CPU Buffer
            // 后才允许 cpu_weight 被销毁。
            const cudaError_t err = cudaDeviceSynchronize();

            if (err != cudaSuccess)
                throw std::runtime_error(
                    cudaGetErrorString(err));
            // printf(
            //     "Loaded weight: %s\n",
            //     name.c_str());
        };

        // ===========================================
        // 1. Token Embedding
        // ===========================================

        with_weight(
            "model.embed_tokens.weight",
            {V, H},
            [&](const Tensor &t)
            {
                model.embedding().load_weight(t);
            });

        // ===========================================
        // 2. 所有 TransformerBlock
        // ===========================================

        for (std::size_t i = 0;
             i < config.num_layers;
             ++i)
        {
            auto &block = model.block(i);

            const std::string prefix =
                "model.layers." + std::to_string(i) + ".";

            // RMSNorm1
            with_weight(
                prefix + "input_layernorm.weight",
                {H},
                [&](const Tensor &t)
                {
                    block.input_norm().load_weight(t);
                });

            // Attention Q
            with_weight(
                prefix + "self_attn.q_proj.weight",
                {QD, H},
                [&](const Tensor &t)
                {
                    block.attention()
                        .q_proj()
                        .load_weight(t);
                });

            // Attention K
            with_weight(
                prefix + "self_attn.k_proj.weight",
                {KVD, H},
                [&](const Tensor &t)
                {
                    block.attention()
                        .k_proj()
                        .load_weight(t);
                });

            // Attention V
            with_weight(
                prefix + "self_attn.v_proj.weight",
                {KVD, H},
                [&](const Tensor &t)
                {
                    block.attention()
                        .v_proj()
                        .load_weight(t);
                });

            // Attention Output Projection
            with_weight(
                prefix + "self_attn.o_proj.weight",
                {H, QD},
                [&](const Tensor &t)
                {
                    block.attention()
                        .o_proj()
                        .load_weight(t);
                });

            // RMSNorm2
            with_weight(
                prefix + "post_attention_layernorm.weight",
                {H},
                [&](const Tensor &t)
                {
                    block.post_attention_norm()
                        .load_weight(t);
                });

            // SwiGLU Gate
            with_weight(
                prefix + "mlp.gate_proj.weight",
                {I, H},
                [&](const Tensor &t)
                {
                    block.feed_forward()
                        .gate_proj()
                        .load_weight(t);
                });

            // SwiGLU Up
            with_weight(
                prefix + "mlp.up_proj.weight",
                {I, H},
                [&](const Tensor &t)
                {
                    block.feed_forward()
                        .up_proj()
                        .load_weight(t);
                });

            // SwiGLU Down
            with_weight(
                prefix + "mlp.down_proj.weight",
                {H, I},
                [&](const Tensor &t)
                {
                    block.feed_forward()
                        .down_proj()
                        .load_weight(t);
                });
        }

        // ===========================================
        // 3. Final RMSNorm
        // ===========================================

        with_weight(
            "model.norm.weight",
            {H},
            [&](const Tensor &t)
            {
                model.final_norm().load_weight(t);
            });

        // ===========================================
        // 4. LM Head
        // ===========================================

        if (!config.tie_word_embeddings)
        {
            with_weight(
                "lm_head.weight",
                {V, H},
                [&](const Tensor &t)
                {
                    model.load_lm_head_weight(t);
                });
        }
    }

} // namespace minillm
