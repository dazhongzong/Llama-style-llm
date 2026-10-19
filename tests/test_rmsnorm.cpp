#include <cmath>
#include <iostream>
#include <stdexcept>

#include "minillm/ops/rmsnorm.h"

namespace
{

void test_rmsnorm_cpu_1d()
{
    using namespace minillm;

    constexpr std::size_t H = 4;

    Tensor input({H}, DataType::FLOAT32, DeviceType::CPU);
    Tensor weight({H}, DataType::FLOAT32, DeviceType::CPU);
    Tensor output({H}, DataType::FLOAT32, DeviceType::CPU);

    for (std::size_t i = 0; i < H; ++i)
    {
        input.ptr<float>()[i] = static_cast<float>(i + 1);
        weight.ptr<float>()[i] = 1.0f;
    }

    rmsnorm_cpu(input, weight, output);

    float sum_sq = 0.0f;
    for (std::size_t i = 0; i < H; ++i)
    {
        sum_sq += input.ptr<float>()[i] * input.ptr<float>()[i];
    }

    const float inv_rms =
        1.0f / std::sqrt(sum_sq / static_cast<float>(H) + 1e-5f);

    for (std::size_t i = 0; i < H; ++i)
    {
        const float expected = input.ptr<float>()[i] * inv_rms;

        if (std::fabs(output.ptr<float>()[i] - expected) > 1e-5f)
        {
            throw std::runtime_error(
                "RMSNorm 1D CPU result mismatch");
        }
    }
}

} // namespace

int main()
{
    using namespace minillm;

    constexpr std::size_t ROWS = 2;
    constexpr std::size_t H = 4;

    Tensor h_x({ROWS, H}, DataType::FLOAT32,
               DeviceType::CPU);

    Tensor h_w({H}, DataType::FLOAT32,
               DeviceType::CPU);

    Tensor h_expected({ROWS, H}, DataType::FLOAT32,
                      DeviceType::CPU);

    Tensor h_result({ROWS, H}, DataType::FLOAT32,
                    DeviceType::CPU);

    float values[8] = {
        1, 2, 3, 4,
        2, 2, 2, 2};

    for (std::size_t i = 0; i < 8; ++i)
    {
        h_x.ptr<float>()[i] = values[i];
    }

    for (std::size_t i = 0; i < H; ++i)
    {
        h_w.ptr<float>()[i] = 1.0f;
    }

    // CPU Reference
    rmsnorm_cpu(h_x, h_w, h_expected);

    // GPU Tensors
    Tensor d_x({ROWS, H}, DataType::FLOAT32,
               DeviceType::CUDA);

    Tensor d_w({H}, DataType::FLOAT32,
               DeviceType::CUDA);

    Tensor d_y({ROWS, H}, DataType::FLOAT32,
               DeviceType::CUDA);

    d_x.copy_from(h_x);
    d_w.copy_from(h_w);

    // GPU 异步执行，然后显式等待默认 stream 完成
    rmsnorm_cuda_async(
        d_x,
        d_w,
        d_y,
        1e-5f,
        nullptr);

    const cudaError_t sync_error =
        cudaStreamSynchronize(nullptr);

    if (sync_error != cudaSuccess)
    {
        throw std::runtime_error(
            cudaGetErrorString(sync_error));
    }

    // GPU → CPU
    h_result.copy_from(d_y);

    // 比较 CPU / CUDA 结果
    for (std::size_t i = 0; i < 8; ++i)
    {
        const float expected =
            h_expected.ptr<float>()[i];

        const float actual =
            h_result.ptr<float>()[i];

        const float tolerance =
            1e-4f + 1e-4f * std::fabs(expected);

        if (!std::isfinite(actual) ||
            std::fabs(actual - expected) > tolerance)
        {
            throw std::runtime_error(
                "RMSNorm result mismatch");
        }

        std::cout << actual << " ";
    }

    test_rmsnorm_cpu_1d();

    std::cout << "\nRMSNorm 1D/2D test passed!\n";
}
