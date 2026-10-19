
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "minillm/model/transformer_model.h"

namespace
{

    void zero(minillm::Tensor &tensor)
    {
        std::fill_n(
            tensor.ptr<float>(),
            tensor.numel(),
            0.0f);
    }

    void ones(minillm::Tensor &tensor)
    {
        std::fill_n(
            tensor.ptr<float>(),
            tensor.numel(),
            1.0f);
    }

    void check_close(float actual, float expected)
    {
        const float allowed =
            1e-4f * (1.0f + std::fabs(expected));

        if (!std::isfinite(actual) ||
            std::fabs(actual - expected) > allowed)
        {
            throw std::runtime_error(
                "TransformerModel logits mismatch");
        }
    }

} // namespace

int main()
{
    using namespace minillm;

    ModelConfig config = ModelConfig::toy();

    TransformerModel model(config);
    DecodeSession session(config);
    ModelWorkspace workspace(config);

    const std::size_t H = config.hidden_size;
    const std::size_t V = config.vocab_size;
    const std::size_t I = config.intermediate_size;
    const std::size_t QD =
        config.num_q_heads * config.head_dim;
    const std::size_t KVD =
        config.num_kv_heads * config.head_dim;

    // ========================================
    // 1. 初始化 Embedding Weight [V,H]
    // ========================================
    Tensor embedding_weight(
        {V, H},
        DataType::FLOAT32,
        DeviceType::CPU);

    zero(embedding_weight);

    // Token 0～3：2 倍单位向量
    for (std::size_t i = 0; i < H; ++i)
    {
        embedding_weight.ptr<float>()[i * H + i] = 2.0f;
    }

    model.embedding().load_weight(embedding_weight);

    // ========================================
    // 2. 初始化 RMSNorm Gamma [H]
    // ========================================
    Tensor gamma(
        {H},
        DataType::FLOAT32,
        DeviceType::CPU);

    ones(gamma);

    model.final_norm().load_weight(gamma);

    // ========================================
    // 3. 构造全零 QKV/O/FFN 权重
    // ========================================
    Tensor wq(
        {QD, H},
        DataType::FLOAT32,
        DeviceType::CPU);

    Tensor wk(
        {KVD, H},
        DataType::FLOAT32,
        DeviceType::CPU);

    Tensor wv(
        {KVD, H},
        DataType::FLOAT32,
        DeviceType::CPU);

    Tensor wo(
        {H, QD},
        DataType::FLOAT32,
        DeviceType::CPU);

    Tensor wg(
        {I, H},
        DataType::FLOAT32,
        DeviceType::CPU);

    Tensor wu(
        {I, H},
        DataType::FLOAT32,
        DeviceType::CPU);

    Tensor wd(
        {H, I},
        DataType::FLOAT32,
        DeviceType::CPU);

    zero(wq);
    zero(wk);
    zero(wv);
    zero(wo);
    zero(wg);
    zero(wu);
    zero(wd);

    // ========================================
    // 4. 初始化两层 TransformerBlock
    // ========================================
    for (std::size_t layer = 0;
         layer < config.num_layers;
         ++layer)
    {
        auto &block = model.block(layer);

        block.input_norm().load_weight(gamma);
        block.post_attention_norm().load_weight(gamma);

        block.attention().q_proj().load_weight(wq);
        block.attention().k_proj().load_weight(wk);
        block.attention().v_proj().load_weight(wv);
        block.attention().o_proj().load_weight(wo);

        block.feed_forward().gate_proj().load_weight(wg);
        block.feed_forward().up_proj().load_weight(wu);
        block.feed_forward().down_proj().load_weight(wd);
    }

    // ========================================
    // 5. CPU Logits Buffer
    // ========================================
    Tensor h_logits(
        {V},
        DataType::FLOAT32,
        DeviceType::CPU);

    // 连续处理 Token 1 和 Token 2
    const std::vector<std::size_t> tokens = {1, 2};

    for (std::size_t pos = 0;
         pos < tokens.size();
         ++pos)
    {
        const std::size_t token_id = tokens[pos];

        // 完整 GPU Decode
        model.forward_decode(
            token_id,
            pos,
            session,
            workspace);

        // GPU → CPU
        h_logits.copy_from(workspace.logits);

        // expected logit[token_id] = 4/sqrt(1+eps)
        const float expected =
            4.0f / std::sqrt(
                       1.0f + config.rms_norm_eps);

        for (std::size_t v = 0; v < V; ++v)
        {
            const float expected_v =
                (v == token_id) ? expected : 0.0f;

            check_close(
                h_logits.ptr<float>()[v],
                expected_v);
        }

        // Argmax 应该等于当前选取的 Token ID
        const float *begin = h_logits.ptr<float>();
        const float *end = begin + V;

        const std::size_t predicted =
            static_cast<std::size_t>(
                std::max_element(begin, end) - begin);

        if (predicted != token_id)
        {
            throw std::runtime_error(
                "Incorrect greedy token");
        }

        // 所有 Layer 的 KV Cache 长度都应当相同
        for (std::size_t layer = 0;
             layer < config.num_layers;
             ++layer)
        {
            if (session.cache(layer).size() != pos + 1)
            {
                throw std::runtime_error(
                    "Per-layer KV Cache mismatch");
            }
        }

        std::cout
            << "[PASS] Model token position "
            << pos << '\n';
    }

    std::cout
        << "All TransformerModel tests passed!\n";

    return 0;
}
