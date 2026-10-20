
#include "minillm/model/model_config_loader.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <stdexcept>

namespace minillm
{

    ModelConfig load_model_config(
        const std::filesystem::path &config_path,
        std::size_t kv_capacity)
    {
        std::ifstream file(config_path);

        if (!file)
        {
            throw std::runtime_error(
                "Cannot open model config");
        }

        nlohmann::json j;
        file >> j;

        if (j.value("model_type", std::string{}) != "llama")
        {
            throw std::runtime_error(
                "Only LLaMA-like model is supported");
        }

        if (j.value("hidden_act", std::string{}) != "silu" ||
            j.value("attention_bias", false) ||
            j.value("mlp_bias", false) ||
            (j.contains("rope_scaling") &&
             !j.at("rope_scaling").is_null()))
        {
            throw std::runtime_error(
                "Unsupported model architecture options");
        }

        ModelConfig c;

        c.vocab_size =
            j.at("vocab_size").get<std::size_t>();

        c.hidden_size =
            j.at("hidden_size").get<std::size_t>();

        c.intermediate_size =
            j.at("intermediate_size").get<std::size_t>();

        c.num_layers =
            j.at("num_hidden_layers").get<std::size_t>();

        c.num_q_heads =
            j.at("num_attention_heads").get<std::size_t>();

        c.num_kv_heads =
            j.at("num_key_value_heads").get<std::size_t>();

        c.max_position_embeddings =
            j.at("max_position_embeddings").get<std::size_t>();

        if (c.num_q_heads == 0 ||
            c.hidden_size % c.num_q_heads != 0)
        {
            throw std::runtime_error(
                "Cannot infer head_dim");
        }

        c.head_dim = j.contains("head_dim")
                         ? j.at("head_dim").get<std::size_t>()
                         : c.hidden_size / c.num_q_heads;

        c.rms_norm_eps =
            j.value("rms_norm_eps", 1e-5f);

        c.rope_theta =
            j.value("rope_theta", 10000.0f);

        const bool interleaved =
            j.value("rope_interleaved", false);

        c.rope_layout = interleaved
                            ? RoPELayout::Interleaved
                            : RoPELayout::HalfSplit;

        c.tie_word_embeddings =
            j.value("tie_word_embeddings", false);

        c.kv_capacity = kv_capacity;

        c.validate();

        // 当前实现沿用 Q 的总维度等于 Hidden Size。
        if (c.num_q_heads * c.head_dim != c.hidden_size)
        {
            throw std::runtime_error(
                "Unsupported Q projection dimension");
        }

        return c;
    }

} // namespace minillm
