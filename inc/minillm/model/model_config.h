
#pragma once

#include <cstddef>
#include <stdexcept>

namespace minillm
{

    enum class RoPELayout
    {
        Interleaved, // 相邻元素配对：当前教学版
        HalfSplit    // 前后半维配对：真实 LLaMA 常见形式
    };

    struct ModelConfig
    {
        std::size_t vocab_size = 0;
        std::size_t hidden_size = 0;
        std::size_t intermediate_size = 0;

        std::size_t num_layers = 0;
        std::size_t num_q_heads = 0;
        std::size_t num_kv_heads = 0;
        std::size_t head_dim = 0;

        std::size_t max_position_embeddings = 0;
        std::size_t kv_capacity = 0;

        float rms_norm_eps = 1e-5f;
        float rope_theta = 10000.0f;

        RoPELayout rope_layout = RoPELayout::Interleaved;
        bool tie_word_embeddings = true;

        void validate() const
        {
            if (vocab_size == 0 ||
                hidden_size == 0 ||
                intermediate_size == 0 ||
                num_layers == 0 ||
                num_q_heads == 0 ||
                num_kv_heads == 0 ||
                head_dim == 0 ||
                head_dim % 2 != 0 ||
                num_q_heads % num_kv_heads != 0 ||
                kv_capacity == 0 ||
                max_position_embeddings < kv_capacity ||
                rms_norm_eps <= 0.0f ||
                rope_theta <= 0.0f)
            {
                throw std::invalid_argument(
                    "Invalid ModelConfig");
            }
        }

        static ModelConfig toy()
        {
            ModelConfig c;
            c.vocab_size = 8;
            c.hidden_size = 4;
            c.intermediate_size = 8;
            c.num_layers = 2;
            c.num_q_heads = 2;
            c.num_kv_heads = 1;
            c.head_dim = 2;
            c.max_position_embeddings = 8;
            c.kv_capacity = 8;
            c.tie_word_embeddings = true;
            return c;
        }
    };

} // namespace minillm
