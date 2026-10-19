
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "minillm/layers/transformer_block.h"

namespace
{

    constexpr std::size_t H = 4;
    constexpr std::size_t I = 8;
    constexpr std::size_t Hq = 2;
    constexpr std::size_t Hkv = 1;
    constexpr std::size_t D = 2;
    constexpr std::size_t Capacity = 4;
    constexpr float Eps = 1e-5f;

    using Vec4 = std::array<float, H>;
    using Vec2 = std::array<float, D>;

    // 两个 RMSNorm 使用不同的权重
    const Vec4 Gamma1 = {1.0f, 0.8f, 1.1f, 0.9f};
    const Vec4 Gamma2 = {0.9f, 1.1f, 0.95f, 1.05f};

    Vec4 rmsnorm_ref(
        const Vec4 &x,
        const Vec4 &gamma)
    {
        double sum_sq = 0.0;

        for (float v : x)
            sum_sq += double(v) * double(v);

        const double inv_rms =
            1.0 / std::sqrt(sum_sq / H + Eps);

        Vec4 y{};

        for (std::size_t i = 0; i < H; ++i)
            y[i] = float(x[i] * inv_rms * gamma[i]);

        return y;
    }

    Vec2 rope_ref(Vec2 v, std::size_t position)
    {
        // 本测试 D=2，沿用相邻元素配对的 RoPE
        const float angle = float(position);
        const float c = std::cos(angle);
        const float s = std::sin(angle);

        return {
            v[0] * c - v[1] * s,
            v[0] * s + v[1] * c};
    }

    float silu_ref(float x)
    {
        return x / (1.0f + std::exp(-x));
    }

    struct CpuBlockReference
    {
        std::vector<Vec2> keys;
        std::vector<Vec2> values;

        Vec4 forward(
            const Vec4 &x,
            std::size_t position)
        {
            if (position != keys.size())
                throw std::runtime_error(
                    "CPU reference position mismatch");

            // 1. RMSNorm1
            const Vec4 n1 = rmsnorm_ref(x, Gamma1);

            // 2. QKV Projection
            // Wq = I4
            // Wk 取 n1[0], n1[1]
            // Wv 取 n1[2], n1[3]

            Vec2 q0 = {n1[0], n1[1]};
            Vec2 q1 = {n1[2], n1[3]};

            Vec2 k = {n1[0], n1[1]};
            Vec2 v = {n1[2], n1[3]};

            // 3. RoPE
            q0 = rope_ref(q0, position);
            q1 = rope_ref(q1, position);
            k = rope_ref(k, position);

            // 4. KV Cache
            keys.push_back(k);
            values.push_back(v);

            // 5. Attention（GQA：两个 Q Head 共享一个 KV Head）
            Vec4 context{};

            const std::array<Vec2, 2> queries = {q0, q1};

            for (std::size_t head = 0; head < Hq; ++head)
            {
                std::vector<double> scores(keys.size());

                for (std::size_t t = 0; t < keys.size(); ++t)
                {
                    const double dot =
                        double(queries[head][0]) * keys[t][0] +
                        double(queries[head][1]) * keys[t][1];

                    scores[t] = dot / std::sqrt(double(D));
                }

                const double m =
                    *std::max_element(scores.begin(), scores.end());

                double denom = 0.0;

                for (double &score : scores)
                {
                    score = std::exp(score - m);
                    denom += score;
                }

                for (std::size_t t = 0; t < keys.size(); ++t)
                {
                    const double p = scores[t] / denom;

                    context[head * D + 0] +=
                        float(p * values[t][0]);

                    context[head * D + 1] +=
                        float(p * values[t][1]);
                }
            }

            // Wo = I4，Context 就是 Attention 投影输出

            // 6. Residual Add1
            Vec4 residual{};

            for (std::size_t i = 0; i < H; ++i)
                residual[i] = x[i] + context[i];

            // 7. RMSNorm2
            const Vec4 n2 = rmsnorm_ref(residual, Gamma2);

            // 8. FFN
            //
            // Gate 前 4 维 = n2
            // Gate 后 4 维 = 0.5*n2
            //
            // Up 前 4 维 = 0.5*n2
            // Up 后 4 维 = n2
            //
            // Down 对两组激活加权：
            // ffn[i] = 0.2*act[i] + 0.1*act[i+4]

            Vec4 ffn_out{};

            for (std::size_t i = 0; i < H; ++i)
            {
                const float act0 =
                    silu_ref(n2[i]) * (0.5f * n2[i]);

                const float act1 =
                    silu_ref(0.5f * n2[i]) * n2[i];

                ffn_out[i] =
                    0.2f * act0 + 0.1f * act1;
            }

            // 9. Residual Add2
            Vec4 output{};

            for (std::size_t i = 0; i < H; ++i)
                output[i] = residual[i] + ffn_out[i];

            return output;
        }
    };

    void fill_zero(minillm::Tensor &t)
    {
        std::fill_n(t.ptr<float>(), t.numel(), 0.0f);
    }

    void check_close(
        float actual,
        float expected,
        std::size_t position,
        std::size_t index)
    {
        const float tolerance =
            3e-4f * (1.0f + std::fabs(expected));

        if (!std::isfinite(actual) ||
            std::fabs(actual - expected) > tolerance)
        {
            throw std::runtime_error(
                "Mismatch at token=" +
                std::to_string(position) +
                ", index=" +
                std::to_string(index) +
                ", expected=" +
                std::to_string(expected) +
                ", actual=" +
                std::to_string(actual));
        }
    }

} // namespace

int main()
{
    using namespace minillm;

    TransformerBlock model(H, I, Hq, Hkv, D, Eps);

    TransformerBlockWorkspace workspace(
        H, I, Hq, Hkv, D, Capacity);

    CudaKVCache cache(Capacity, Hkv, D);

    // ------------------------------------------
    // 1. RMSNorm Gamma
    // ------------------------------------------

    Tensor gamma1({H}, DataType::FLOAT32, DeviceType::CPU);
    Tensor gamma2({H}, DataType::FLOAT32, DeviceType::CPU);

    std::copy(Gamma1.begin(), Gamma1.end(),
              gamma1.ptr<float>());

    std::copy(Gamma2.begin(), Gamma2.end(),
              gamma2.ptr<float>());

    model.input_norm().load_weight(gamma1);

    model.post_attention_norm().load_weight(gamma2);

    // ------------------------------------------
    // 2. QKV / Output Linear 权重
    // ------------------------------------------

    Tensor wq({H, H}, DataType::FLOAT32, DeviceType::CPU);
    Tensor wk({Hkv * D, H}, DataType::FLOAT32, DeviceType::CPU);
    Tensor wv({Hkv * D, H}, DataType::FLOAT32, DeviceType::CPU);
    Tensor wo({H, H}, DataType::FLOAT32, DeviceType::CPU);

    fill_zero(wq);
    fill_zero(wk);
    fill_zero(wv);
    fill_zero(wo);

    // Wq = I4，Wo = I4
    for (std::size_t i = 0; i < H; ++i)
    {
        wq.ptr<float>()[i * H + i] = 1.0f;
        wo.ptr<float>()[i * H + i] = 1.0f;
    }

    // Wk 取输入前两维
    wk.ptr<float>()[0 * H + 0] = 1.0f;
    wk.ptr<float>()[1 * H + 1] = 1.0f;

    // Wv 取输入后两维
    wv.ptr<float>()[0 * H + 2] = 1.0f;
    wv.ptr<float>()[1 * H + 3] = 1.0f;

    model.attention().q_proj().load_weight(wq);
    model.attention().k_proj().load_weight(wk);
    model.attention().v_proj().load_weight(wv);
    model.attention().o_proj().load_weight(wo);

    // ------------------------------------------
    // 3. FeedForward 权重
    // ------------------------------------------

    Tensor wg({I, H}, DataType::FLOAT32, DeviceType::CPU);
    Tensor wu({I, H}, DataType::FLOAT32, DeviceType::CPU);
    Tensor wd({H, I}, DataType::FLOAT32, DeviceType::CPU);

    fill_zero(wg);
    fill_zero(wu);
    fill_zero(wd);

    for (std::size_t i = 0; i < H; ++i)
    {
        // Gate [8,4]
        wg.ptr<float>()[i * H + i] = 1.0f;
        wg.ptr<float>()[(i + H) * H + i] = 0.5f;

        // Up [8,4]
        wu.ptr<float>()[i * H + i] = 0.5f;
        wu.ptr<float>()[(i + H) * H + i] = 1.0f;

        // Down [4,8]
        wd.ptr<float>()[i * I + i] = 0.2f;
        wd.ptr<float>()[i * I + i + H] = 0.1f;
    }

    model.feed_forward().gate_proj().load_weight(wg);
    model.feed_forward().up_proj().load_weight(wu);
    model.feed_forward().down_proj().load_weight(wd);

    // ------------------------------------------
    // 4. 输入与输出
    // ------------------------------------------

    Tensor h_input({H}, DataType::FLOAT32, DeviceType::CPU);
    Tensor h_output({H}, DataType::FLOAT32, DeviceType::CPU);

    Tensor d_input({H}, DataType::FLOAT32, DeviceType::CUDA);
    Tensor d_output({H}, DataType::FLOAT32, DeviceType::CUDA);

    CpuBlockReference cpu_ref;

    const std::array<Vec4, 2> tokens = {{{{1.0f, 0.0f, 3.0f, 5.0f}},
                                         {{0.0f, 1.0f, 7.0f, 11.0f}}}};

    // ------------------------------------------
    // 5. 连续 Decode 两个 Token
    // ------------------------------------------

    for (std::size_t pos = 0; pos < tokens.size(); ++pos)
    {
        const Vec4 &input = tokens[pos];

        // CPU Reference
        const Vec4 expected = cpu_ref.forward(input, pos);

        // H2D
        std::copy(input.begin(), input.end(),
                  h_input.ptr<float>());

        d_input.copy_from(h_input);

        // GPU TransformerBlock，内部同步
        model.forward(
            d_input,
            d_output,
            cache,
            workspace,
            pos);

        // D2H
        h_output.copy_from(d_output);

        if (cache.size() != pos + 1)
            throw std::runtime_error(
                "KV Cache length mismatch");

        for (std::size_t i = 0; i < H; ++i)
        {
            check_close(
                h_output.ptr<float>()[i],
                expected[i],
                pos,
                i);
        }

        std::cout
            << "[PASS] TransformerBlock Token "
            << pos << '\n';
    }

    std::cout
        << "All TransformerBlock tests passed!\n";
}
