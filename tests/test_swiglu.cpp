#include <cmath>
#include <iostream>
#include <stdexcept>

#include "minillm/ops/swiglu.h"

int main()
{
    using namespace minillm;

    constexpr std::size_t N = 1003;

    Tensor h_gate(
        {N}, DataType::FLOAT32, DeviceType::CPU);

    Tensor h_up(
        {N}, DataType::FLOAT32, DeviceType::CPU);

    Tensor h_expected(
        {N}, DataType::FLOAT32, DeviceType::CPU);

    Tensor h_actual(
        {N}, DataType::FLOAT32, DeviceType::CPU);

    Tensor d_gate(
        {N}, DataType::FLOAT32, DeviceType::CUDA);

    Tensor d_up(
        {N}, DataType::FLOAT32, DeviceType::CUDA);

    Tensor d_output(
        {N}, DataType::FLOAT32, DeviceType::CUDA);

    // 1. 初始化输入
    for (std::size_t i = 0; i < N; ++i)
    {
        h_gate.ptr<float>()[i] =
            0.1f * static_cast<float>(
                       static_cast<int>(i % 83) - 41);

        h_up.ptr<float>()[i] =
            0.05f * static_cast<float>(
                        static_cast<int>(i % 13) - 6);
    }

    // 2. CPU Reference
    swiglu_cpu(
        h_gate,
        h_up,
        h_expected);

    // 3. H2D
    d_gate.copy_from(h_gate);
    d_up.copy_from(h_up);

    // 4. CUDA 计算
    swiglu_cuda(
        d_gate,
        d_up,
        d_output);

    // 5. D2H
    h_actual.copy_from(d_output);

    // 6. 验证
    float max_error = 0.0f;

    for (std::size_t i = 0; i < N; ++i)
    {
        const float expected =
            h_expected.ptr<float>()[i];

        const float actual =
            h_actual.ptr<float>()[i];

        if (!std::isfinite(actual))
        {
            throw std::runtime_error(
                "SwiGLU produced NaN or Inf");
        }

        const float error =
            std::fabs(actual - expected);

        max_error = std::fmax(max_error, error);

        const float tolerance =
            1e-5f + 1e-4f * std::fabs(expected);

        if (error > tolerance)
        {
            throw std::runtime_error(
                "CPU/CUDA SwiGLU mismatch");
        }
    }

    std::cout
        << "[PASS] CUDA SwiGLU\n"
        << "Max absolute error: "
        << max_error << '\n';
}