#include <cmath>
#include <iostream>
#include <stdexcept>

#include "minillm/ops/rope.h"

int main()
{
    using namespace minillm;

    constexpr std::size_t S = 3;
    constexpr std::size_t H = 2;
    constexpr std::size_t D = 8;

    Tensor h_input(
        {S, H, D}, DataType::FLOAT32, DeviceType::CPU);

    Tensor h_expected(
        {S, H, D}, DataType::FLOAT32, DeviceType::CPU);

    Tensor h_actual(
        {S, H, D}, DataType::FLOAT32, DeviceType::CPU);

    Tensor d_input(
        {S, H, D}, DataType::FLOAT32, DeviceType::CUDA);

    Tensor d_output(
        {S, H, D}, DataType::FLOAT32, DeviceType::CUDA);

    for (std::size_t i = 0; i < S * H * D; ++i)
    {
        h_input.ptr<float>()[i] =
            0.1f * static_cast<float>(
                       static_cast<int>(i % 19) - 9);
    }

    // 非零起始位置，模拟 Decode
    constexpr std::size_t start_pos = 7;

    rope_cpu(h_input, h_expected, start_pos);

    d_input.copy_from(h_input);

    rope_cuda(d_input, d_output, start_pos);

    h_actual.copy_from(d_output);

    float max_error = 0.0f;

    for (std::size_t i = 0; i < S * H * D; ++i)
    {
        const float expected = h_expected.ptr<float>()[i];
        const float actual = h_actual.ptr<float>()[i];

        const float error = std::fabs(expected - actual);

        max_error = std::fmax(max_error, error);

        const float tolerance =
            1e-4f + 1e-4f * std::fabs(expected);

        if (!std::isfinite(actual) || error > tolerance)
        {
            throw std::runtime_error(
                "CPU/CUDA RoPE mismatch");
        }
    }

    std::cout
        << "[PASS] RoPE\n"
        << "Max error = " << max_error << "\n";
}