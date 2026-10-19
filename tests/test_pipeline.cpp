#include <cmath>
#include <iostream>
#include <stdexcept>

#include "minillm/tensor/tensor.h"
#include "minillm/ops/rmsnorm.h"

using namespace minillm;

void test_rmsnorm_pipeline()
{
    constexpr std::size_t ROWS = 2;
    constexpr std::size_t H = 33;

    // CPU Tensor
    Tensor h_input(
        {ROWS, H},
        DataType::FLOAT32,
        DeviceType::CPU);

    Tensor h_weight(
        {H},
        DataType::FLOAT32,
        DeviceType::CPU);

    Tensor h_expected(
        {ROWS, H},
        DataType::FLOAT32,
        DeviceType::CPU);

    Tensor h_result(
        {ROWS, H},
        DataType::FLOAT32,
        DeviceType::CPU);

    // CUDA Tensor
    Tensor d_input(
        {ROWS, H},
        DataType::FLOAT32,
        DeviceType::CUDA);

    Tensor d_weight(
        {H},
        DataType::FLOAT32,
        DeviceType::CUDA);

    Tensor d_output(
        {ROWS, H},
        DataType::FLOAT32,
        DeviceType::CUDA);

    // 1. 初始化 CPU 输入
    for (std::size_t i = 0; i < ROWS * H; ++i)
    {
        h_input.ptr<float>()[i] =
            static_cast<float>(
                static_cast<int>(i % 9) - 4) *
            0.25f;
    }

    for (std::size_t i = 0; i < H; ++i)
    {
        h_weight.ptr<float>()[i] =
            0.5f + 0.01f * static_cast<float>(i);
    }

    // 2. CPU Reference
    rmsnorm_cpu(
        h_input,
        h_weight,
        h_expected);

    // 3. Host → Device
    d_input.copy_from(h_input);
    d_weight.copy_from(h_weight);

    // 4. CUDA RMSNorm
    rmsnorm_cuda(
        d_input,
        d_weight,
        d_output);

    // 5. Device → Host
    h_result.copy_from(d_output);

    // 6. 比较结果
    float max_error = 0.0f;

    for (std::size_t i = 0; i < ROWS * H; ++i)
    {
        const float expected =
            h_expected.ptr<float>()[i];

        const float actual =
            h_result.ptr<float>()[i];

        if (!std::isfinite(actual))
        {
            throw std::runtime_error(
                "RMSNorm produced NaN or Inf");
        }

        const float error =
            std::fabs(expected - actual);

        if (error > max_error)
        {
            max_error = error;
        }

        const float tolerance =
            1e-4f + 1e-4f * std::fabs(expected);

        if (error > tolerance)
        {
            throw std::runtime_error(
                "CPU/CUDA RMSNorm mismatch");
        }
    }

    std::cout
        << "[PASS] RMSNorm pipeline\n"
        << "Max absolute error: "
        << max_error << '\n';
}

int main()
{
    try
    {
        test_rmsnorm_pipeline();

        std::cout
            << "All pipeline tests passed!\n";

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr
            << "[FAIL] "
            << e.what()
            << '\n';

        return 1;
    }
}