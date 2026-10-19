#include <cmath>
#include <iostream>
#include <stdexcept>

#include "minillm/ops/gemv.h"

int main()
{
    using namespace minillm;

    constexpr std::size_t M = 1003;
    constexpr std::size_t K = 128;

    Tensor h_w({M, K}, DataType::FLOAT32,
               DeviceType::CPU);
    Tensor h_x({K}, DataType::FLOAT32,
               DeviceType::CPU);
    Tensor h_expected({M}, DataType::FLOAT32,
                      DeviceType::CPU);
    Tensor h_actual({M}, DataType::FLOAT32,
                    DeviceType::CPU);

    Tensor d_w({M, K}, DataType::FLOAT32,
               DeviceType::CUDA);
    Tensor d_x({K}, DataType::FLOAT32,
               DeviceType::CUDA);
    Tensor d_y({M}, DataType::FLOAT32,
               DeviceType::CUDA);

    // 初始化权重
    for (std::size_t i = 0; i < M * K; ++i)
    {
        h_w.ptr<float>()[i] =
            static_cast<float>(
                static_cast<int>(i % 17) - 8) *
            0.1f;
    }

    // 初始化输入
    for (std::size_t i = 0; i < K; ++i)
    {
        h_x.ptr<float>()[i] =
            static_cast<float>(
                static_cast<int>(i % 9) - 4) *
            0.2f;
    }

    // CPU Reference
    gemv_cpu(h_w, h_x, h_expected);

    // Host → Device
    d_w.copy_from(h_w);
    d_x.copy_from(h_x);

    // CUDA GEMV
    gemv_cuda(d_w, d_x, d_y);

    // Device → Host
    h_actual.copy_from(d_y);

    float max_error = 0.0f;

    for (std::size_t i = 0; i < M; ++i)
    {
        const float expected =
            h_expected.ptr<float>()[i];

        const float actual =
            h_actual.ptr<float>()[i];

        const float error =
            std::fabs(expected - actual);

        max_error = std::fmax(max_error, error);

        const float tolerance =
            1e-4f + 1e-4f * std::fabs(expected);

        if (!std::isfinite(actual) ||
            error > tolerance)
        {
            throw std::runtime_error(
                "CPU/CUDA GEMV mismatch");
        }
    }

    std::cout
        << "[PASS] CUDA GEMV\n"
        << "Maximum absolute error: "
        << max_error << '\n';
}