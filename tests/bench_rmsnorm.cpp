#include <iostream>

#include "minillm/ops/rmsnorm.h"

int main()
{
    using namespace minillm;

    constexpr std::size_t ROWS = 8;
    constexpr std::size_t H = 4096;

    Tensor h_x({ROWS, H},
               DataType::FLOAT32, DeviceType::CPU);

    Tensor h_w({H},
               DataType::FLOAT32, DeviceType::CPU);

    Tensor d_x({ROWS, H},
               DataType::FLOAT32, DeviceType::CUDA);

    Tensor d_w({H},
               DataType::FLOAT32, DeviceType::CUDA);

    Tensor d_y({ROWS, H},
               DataType::FLOAT32, DeviceType::CUDA);

    for (std::size_t i = 0; i < ROWS * H; ++i)
    {
        h_x.ptr<float>()[i] =
            static_cast<float>(
                static_cast<int>(i % 23) - 11) *
            0.1f;
    }

    for (std::size_t i = 0; i < H; ++i)
    {
        h_w.ptr<float>()[i] =
            0.5f + 0.01f * static_cast<float>(i % 17);
    }

    d_x.copy_from(h_x);
    d_w.copy_from(h_w);

    float shared_us = benchmark_rmsnorm_cuda(
        d_x, d_w, d_y, RMSNormVersion::Shared);

    float warp_us = benchmark_rmsnorm_cuda(
        d_x, d_w, d_y, RMSNormVersion::Warp);

    std::cout << "Shape: ["
              << ROWS << ", " << H << "]\n";

    std::cout << "Shared: "
              << shared_us << " us\n";

    std::cout << "Warp:   "
              << warp_us << " us\n";

    std::cout << "Speedup (Shared/Warp): "
              << shared_us / warp_us << "x\n";
}