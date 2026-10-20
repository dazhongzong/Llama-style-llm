#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "minillm/layers/self_attention.h"

namespace
{

    void check_close(
        float actual,
        float expected,
        float tolerance = 1e-4f)
    {
        const float allowed =
            tolerance * (1.0f + std::fabs(expected));

        if (!std::isfinite(actual) ||
            std::fabs(actual - expected) > allowed)
        {
            throw std::runtime_error(
                "SelfAttention result mismatch");
        }
    }

} // namespace

int main()
{
    using namespace minillm;

    constexpr std::size_t H = 4;
    constexpr std::size_t Hq = 2;
    constexpr std::size_t Hkv = 1;
    constexpr std::size_t D = 2;
    constexpr std::size_t C = 4;

    // 1. 创建模块、Workspace 和 KV Cache
    SelfAttention model(H, Hq, Hkv, D, RoPEConfig{10000.0f, RoPELayout::Interleaved});

    SelfAttentionWorkspace workspace(
        Hq, Hkv, D, C);

    CudaKVCache cache(C, Hkv, D);

    // 2. CPU 权重
    Tensor wq({4, 4}, DataType::FLOAT32, DeviceType::CPU);
    Tensor wk({2, 4}, DataType::FLOAT32, DeviceType::CPU);
    Tensor wv({2, 4}, DataType::FLOAT32, DeviceType::CPU);
    Tensor wo({4, 4}, DataType::FLOAT32, DeviceType::CPU);

    std::fill_n(wq.ptr<float>(), wq.numel(), 0.0f);
    std::fill_n(wk.ptr<float>(), wk.numel(), 0.0f);
    std::fill_n(wv.ptr<float>(), wv.numel(), 0.0f);
    std::fill_n(wo.ptr<float>(), wo.numel(), 0.0f);

    // Wq = I4, Wo = I4
    for (std::size_t i = 0; i < H; ++i)
    {
        wq.ptr<float>()[i * H + i] = 1.0f;
        wo.ptr<float>()[i * H + i] = 1.0f;
    }

    // Wk：取 x[0], x[1]
    wk.ptr<float>()[0 * H + 0] = 1.0f;
    wk.ptr<float>()[1 * H + 1] = 1.0f;

    // Wv：取 x[2], x[3]
    wv.ptr<float>()[0 * H + 2] = 1.0f;
    wv.ptr<float>()[1 * H + 3] = 1.0f;

    // 3. 加载权重到 GPU Linear
    model.q_proj().load_weight(wq);
    model.k_proj().load_weight(wk);
    model.v_proj().load_weight(wv);
    model.o_proj().load_weight(wo);

    // 4. 输入/输出 Tensor
    Tensor h_input(
        {H}, DataType::FLOAT32, DeviceType::CPU);

    Tensor h_output(
        {H}, DataType::FLOAT32, DeviceType::CPU);

    Tensor d_input(
        {H}, DataType::FLOAT32, DeviceType::CUDA);

    Tensor d_output(
        {H}, DataType::FLOAT32, DeviceType::CUDA);

    auto run_token = [&](
                         std::size_t position,
                         const std::array<float, 4> &data)
    {
        std::copy(
            data.begin(),
            data.end(),
            h_input.ptr<float>());

        // CPU → GPU
        d_input.copy_from(h_input);

        // QKV + RoPE + Cache + Attention + Wo
        model.forward(
            d_input,
            d_output,
            cache,
            workspace,
            position);

        // GPU → CPU
        h_output.copy_from(d_output);

        if (cache.size() != position + 1)
        {
            throw std::runtime_error(
                "Incorrect KV Cache length");
        }
    };

    // ================================================
    // 测试 Token 0
    // ================================================

    run_token(
        0,
        {1.0f, 0.0f, 3.0f, 5.0f});

    const std::array<float, 4> expected0 = {
        3.0f, 5.0f, 3.0f, 5.0f};

    for (std::size_t i = 0; i < H; ++i)
    {
        check_close(
            h_output.ptr<float>()[i],
            expected0[i]);
    }

    std::cout << "[PASS] SelfAttention Token 0\n";

    // ================================================
    // 测试 Token 1：包含历史缓存和 RoPE
    // ================================================

    run_token(
        1,
        {0.0f, 1.0f, 7.0f, 11.0f});

    const float s = std::sin(1.0f);
    const float c = std::cos(1.0f);
    const float scale = 1.0f / std::sqrt(2.0f);

    // Token 1 的 Q：
    //
    // Q Head 0 = [0,1]
    // Q Head 1 = [7,11]
    //
    // 位置 1 经过相邻元素配对的 RoPE
    const std::array<float, 4> rotated_q = {
        -s, c,
        7.0f * c - 11.0f * s,
        7.0f * s + 11.0f * c};

    // Token 1 的 K = [0,1]
    // 经过 RoPE → [-sin(1), cos(1)]
    const float k1x = -s;
    const float k1y = c;

    // Token 0 的 K = [1,0]
    // 历史 V0 = [3,5]
    // 当前 V1 = [7,11]

    for (std::size_t head = 0; head < Hq; ++head)
    {
        const float qx = rotated_q[head * D];
        const float qy = rotated_q[head * D + 1];

        // 当前 Q 与历史 Key 0 点积
        const float score0 =
            qx * scale;

        // 当前 Q 与 Key 1 点积
        const float score1 =
            (qx * k1x + qy * k1y) * scale;

        // Stable Softmax
        const float max_score =
            std::max(score0, score1);

        const float exp0 =
            std::exp(score0 - max_score);

        const float exp1 =
            std::exp(score1 - max_score);

        const float denom = exp0 + exp1;

        const float p0 = exp0 / denom;
        const float p1 = exp1 / denom;

        // 加权历史 V，得到 Context
        const float expected_x =
            p0 * 3.0f + p1 * 7.0f;

        const float expected_y =
            p0 * 5.0f + p1 * 11.0f;

        // Wo 是单位矩阵，直接检查对应输出
        check_close(
            h_output.ptr<float>()[head * D],
            expected_x);

        check_close(
            h_output.ptr<float>()[head * D + 1],
            expected_y);
    }

    std::cout << "[PASS] SelfAttention Token 1\n";

    // ================================================
    // 测试非法 Position
    // ================================================

    bool position_rejected = false;

    try
    {
        // 当前 Cache size=2，不能直接跳到 position 3
        model.forward(
            d_input,
            d_output,
            cache,
            workspace,
            3);
    }
    catch (const std::invalid_argument &)
    {
        position_rejected = true;
    }

    if (!position_rejected || cache.size() != 2)
    {
        throw std::runtime_error(
            "Position validation failed");
    }

    std::cout << "[PASS] Invalid position rejected\n";

    std::cout
        << "All SelfAttention tests passed!\n";
}