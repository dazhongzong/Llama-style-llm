#include <cmath>
#include <iostream>
#include <stdexcept>

#include "minillm/tensor/tensor.h"
#include "minillm/ops/add.h"
#include "minillm/ops/rmsnorm.h"
using namespace minillm;

void require(bool condition, const char *message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

void test_rmsnorm_cpu()
{
    Tensor input(
        {2, 4},
        DataType::FLOAT32,
        DeviceType::CPU);

    Tensor weight(
        {4},
        DataType::FLOAT32,
        DeviceType::CPU);

    Tensor output(
        {2, 4},
        DataType::FLOAT32,
        DeviceType::CPU);

    float values[8] = {
        1, 2, 3, 4,
        2, 2, 2, 2};

    for (int i = 0; i < 8; ++i)
    {
        input.ptr<float>()[i] = values[i];
    }

    for (int i = 0; i < 4; ++i)
    {
        weight.ptr<float>()[i] = 1.0f;
    }

    rmsnorm_cpu(input, weight, output);

    for (int i = 0; i < 8; ++i)
    {
        std::cout << output.ptr<float>()[i] << " ";
    }
    std::cout << "\n";
}

// 测试 Tensor 基本功能
void test_tensor()
{
    Tensor x(
        {2, 3, 4},
        DataType::FLOAT32,
        DeviceType::CPU);

    require(x.ndim() == 3, "ndim failed");
    require(x.numel() == 24, "numel failed");
    require(x.bytes() == 96, "bytes failed");

    require(
        x.strides()[0] == 12 &&
            x.strides()[1] == 4 &&
            x.strides()[2] == 1,
        "strides failed");

    require(
        x.offset({1, 2, 3}) == 23,
        "offset failed");

    std::cout << "[PASS] Tensor\n";
}

// 测试 CPU → CUDA → CPU
void test_tensor_copy()
{
    constexpr std::size_t N = 8;

    Tensor host_a(
        {N}, DataType::FLOAT32, DeviceType::CPU);

    Tensor device(
        {N}, DataType::FLOAT32, DeviceType::CUDA);

    Tensor host_b(
        {N}, DataType::FLOAT32, DeviceType::CPU);

    for (std::size_t i = 0; i < N; ++i)
    {
        host_a.ptr<float>()[i] =
            static_cast<float>(i + 1);
    }

    device.copy_from(host_a);
    host_b.copy_from(device);

    for (std::size_t i = 0; i < N; ++i)
    {
        require(
            host_b.ptr<float>()[i] ==
                static_cast<float>(i + 1),
            "Tensor copy failed");
    }

    std::cout << "[PASS] Tensor Copy\n";
}

// 测试 CUDA Add
void test_add()
{
    // 故意选择不是 256 整数倍的 N
    constexpr std::size_t N = 1000;

    Tensor h_a(
        {N}, DataType::FLOAT32, DeviceType::CPU);
    Tensor h_b(
        {N}, DataType::FLOAT32, DeviceType::CPU);
    Tensor h_c(
        {N}, DataType::FLOAT32, DeviceType::CPU);

    Tensor d_a(
        {N}, DataType::FLOAT32, DeviceType::CUDA);
    Tensor d_b(
        {N}, DataType::FLOAT32, DeviceType::CUDA);
    Tensor d_c(
        {N}, DataType::FLOAT32, DeviceType::CUDA);

    for (std::size_t i = 0; i < N; ++i)
    {
        h_a.ptr<float>()[i] =
            static_cast<float>(i);

        h_b.ptr<float>()[i] =
            static_cast<float>(i * 2);
    }

    d_a.copy_from(h_a);
    d_b.copy_from(h_b);

    add_cuda(d_a, d_b, d_c);

    h_c.copy_from(d_c);

    for (std::size_t i = 0; i < N; ++i)
    {
        const float expected =
            h_a.ptr<float>()[i] +
            h_b.ptr<float>()[i];

        const float actual =
            h_c.ptr<float>()[i];

        require(
            std::fabs(actual - expected) < 1e-5f,
            "CUDA Add failed");
    }

    std::cout << "[PASS] CUDA Add\n";
}

int main()
{
    try
    {
        test_tensor();
        test_tensor_copy();
        test_add();
        test_rmsnorm_cpu();
        std::cout << "All tests passed!\n";
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr
            << "[FAIL] " << e.what() << '\n';

        return 1;
    }
}