#include "minillm/layers/attention.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace minillm
{

    void attention_decode_cpu(
        const Tensor &q,
        const Tensor &k_new,
        const Tensor &v_new,
        KVCache &cache,
        Tensor &output,
        std::size_t position)
    {
        if (q.ndim() != 2 ||
            k_new.ndim() != 2 ||
            v_new.ndim() != 2 ||
            output.shape() != q.shape())
        {
            throw std::invalid_argument("Invalid attention shape");
        }

        const Tensor *tensors[] = {
            &q,
            &k_new,
            &v_new,
            &output};

        for (const Tensor *t : tensors)
        {
            if (t->device() != DeviceType::CPU ||
                t->dtype() != DataType::FLOAT32)
            {
                throw std::invalid_argument(
                    "Attention requires CPU FP32");
            }
        }

        const std::size_t Hq = q.shape()[0];
        const std::size_t Hkv = cache.num_kv_heads();
        const std::size_t D = cache.head_dim();

        if (Hq == 0 || Hq % Hkv != 0 ||
            q.shape()[1] != D ||
            k_new.shape() != std::vector<std::size_t>{Hkv, D} ||
            v_new.shape() != std::vector<std::size_t>{Hkv, D})
        {
            throw std::invalid_argument(
                "Attention head configuration mismatch");
        }

        // 本次输入的 K/V 必须追加到当前位置
        cache.append(k_new, v_new, position);

        const std::size_t context_len = cache.size();
        const std::size_t group_size = Hq / Hkv;

        const float *query = q.ptr<float>();
        float *result = output.ptr<float>();

        const double scale =
            1.0 / std::sqrt(static_cast<double>(D));

        std::vector<double> scores(context_len);
        std::vector<double> weights(context_len);

        for (std::size_t h = 0; h < Hq; ++h)
        {
            // GQA: 多个 Q Head 共享一个 KV Head
            const std::size_t kv_head = h / group_size;

            const float *q_head = query + h * D;

            // 1. Q 与历史 K 计算点积
            for (std::size_t t = 0; t < context_len; ++t)
            {
                const float *key = cache.key_at(t, kv_head);

                double dot = 0.0;

                for (std::size_t d = 0; d < D; ++d)
                {
                    dot += static_cast<double>(q_head[d]) * key[d];
                }

                scores[t] = dot * scale;
            }

            // 2. Stable Softmax
            const double max_score =
                *std::max_element(scores.begin(), scores.end());

            double sum = 0.0;

            for (std::size_t t = 0; t < context_len; ++t)
            {
                weights[t] = std::exp(scores[t] - max_score);
                sum += weights[t];
            }

            // 3. 对 V 加权求和
            for (std::size_t d = 0; d < D; ++d)
            {
                double acc = 0.0;

                for (std::size_t t = 0; t < context_len; ++t)
                {
                    const float *value =
                        cache.value_at(t, kv_head);

                    acc += (weights[t] / sum) * value[d];
                }

                result[h * D + d] = static_cast<float>(acc);
            }
        }
    }

} // namespace minillm
