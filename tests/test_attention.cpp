#include <cassert>
#include <cmath>
#include <iostream>

#include "minillm/layers/attention.h"

int main()
{
    using namespace minillm;

    KVCache cache(4, 1, 2);

    Tensor q({2, 2}, DataType::FLOAT32, DeviceType::CPU);
    Tensor k({1, 2}, DataType::FLOAT32, DeviceType::CPU);
    Tensor v({1, 2}, DataType::FLOAT32, DeviceType::CPU);
    Tensor out({2, 2}, DataType::FLOAT32, DeviceType::CPU);

    // 两个 Query Head
    q.ptr<float>()[0] = 1.0f;
    q.ptr<float>()[1] = 0.0f;
    q.ptr<float>()[2] = 0.0f;
    q.ptr<float>()[3] = 1.0f;

    // Token 0
    k.ptr<float>()[0] = 1.0f;
    k.ptr<float>()[1] = 0.0f;

    v.ptr<float>()[0] = 2.0f;
    v.ptr<float>()[1] = 4.0f;

    attention_decode_cpu(q, k, v, cache, out, 0);

    for (std::size_t h = 0; h < 2; ++h)
    {
        assert(std::fabs(out.ptr<float>()[h * 2] - 2.0f) < 1e-5f);
        assert(std::fabs(out.ptr<float>()[h * 2 + 1] - 4.0f) < 1e-5f);
    }

    assert(cache.size() == 1);

    // Token 1
    k.ptr<float>()[0] = 0.0f;
    k.ptr<float>()[1] = 1.0f;

    v.ptr<float>()[0] = 6.0f;
    v.ptr<float>()[1] = 8.0f;

    attention_decode_cpu(q, k, v, cache, out, 1);

    assert(cache.size() == 2);

    // 两个 Q Head 对两个 Token 的注意力权重相反
    const double a = std::exp(1.0 / std::sqrt(2.0));
    const double p = a / (a + 1.0);

    const double expected_q0 = 2.0 * p + 6.0 * (1.0 - p);
    const double expected_q1 = 2.0 * (1.0 - p) + 6.0 * p;

    assert(std::fabs(out.ptr<float>()[0] - expected_q0) < 1e-4);
    assert(std::fabs(out.ptr<float>()[2] - expected_q1) < 1e-4);

    std::cout << "[PASS] CPU Attention + KV Cache\n";
}