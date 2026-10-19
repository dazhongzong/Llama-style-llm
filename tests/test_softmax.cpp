#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include "minillm/ops/softmax.h"

using namespace minillm;

namespace
{

    void check_close(
        const Tensor &expected,
        const Tensor &actual,
        const char *name)
    {
        const std::size_t count = expected.numel();
        const float *expected_ptr = expected.ptr<float>();
        const float *actual_ptr = actual.ptr<float>();

        float max_error = 1e-5f;

        for (std::size_t i = 0; i < count; ++i)
        {
            if (!std::isfinite(actual_ptr[i]))
            {
                throw std::runtime_error(
                    std::string(name) + " produced NaN or Inf");
            }

            const float error =
                std::fabs(expected_ptr[i] - actual_ptr[i]);

            max_error = std::fmax(max_error, error);

            const float tolerance =
                2e-5f + 2e-5f * std::fabs(expected_ptr[i]);

            if (error > tolerance)
            {
                throw std::runtime_error(
                    std::string(name) + " does not match CPU reference");
            }
        }

        std::cout
            << "[PASS] " << name
            << ", max error: " << max_error << '\n';
    }

} // namespace

int main()
{
    try
    {
        constexpr std::size_t rows = 3;
        constexpr std::size_t hidden = 100;

        Tensor h_input(
            {rows, hidden},
            DataType::FLOAT32,
            DeviceType::CPU);

        Tensor h_expected(
            {rows, hidden},
            DataType::FLOAT32,
            DeviceType::CPU);

        Tensor h_block(
            {rows, hidden},
            DataType::FLOAT32,
            DeviceType::CPU);

        Tensor h_warp(
            {rows, hidden},
            DataType::FLOAT32,
            DeviceType::CPU);

        Tensor d_input(
            {rows, hidden},
            DataType::FLOAT32,
            DeviceType::CUDA);

        Tensor d_block(
            {rows, hidden},
            DataType::FLOAT32,
            DeviceType::CUDA);

        Tensor d_warp(
            {rows, hidden},
            DataType::FLOAT32,
            DeviceType::CUDA);

        for (std::size_t i = 0; i < rows * hidden; ++i)
        {
            h_input.ptr<float>()[i] =
                static_cast<float>(static_cast<int>(i % 31) - 15) * 0.2f;
        }

        softmax_cpu(h_input, h_expected);

        d_input.copy_from(h_input);

        softmax_cuda(
            d_input,
            d_block,
            SoftmaxKernel::Block);

        softmax_cuda(
            d_input,
            d_warp,
            SoftmaxKernel::Warp);

        h_block.copy_from(d_block);
        h_warp.copy_from(d_warp);

        check_close(h_expected, h_block, "Softmax block kernel");
        check_close(h_expected, h_warp, "Softmax warp kernel");
        check_close(h_block, h_warp, "Softmax block/warp comparison");

        std::cout << "All Softmax tests passed!\n";
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[FAIL] " << e.what() << '\n';
        return 1;
    }
}
