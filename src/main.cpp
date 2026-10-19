#include <cmath>
#include <iostream>
#include <stdexcept>

#include "minillm/ops/add.h"
#include "minillm/ops/rmsnorm.h"
#include "minillm/tensor/tensor.h"

int main()
{
    using namespace minillm;

    constexpr std::size_t N = 8;

    Tensor h_a({N}, DataType::FLOAT32, DeviceType::CPU);
    Tensor h_b({N}, DataType::FLOAT32, DeviceType::CPU);
    Tensor h_c({N}, DataType::FLOAT32, DeviceType::CPU);

    Tensor d_a({N}, DataType::FLOAT32, DeviceType::CUDA);
    Tensor d_b({N}, DataType::FLOAT32, DeviceType::CUDA);
    Tensor d_c({N}, DataType::FLOAT32, DeviceType::CUDA);

    // 1. CPU 初始化
    for (std::size_t i = 0; i < N; ++i)
    {
        h_a.ptr<float>()[i] = static_cast<float>(i);
        h_b.ptr<float>()[i] = static_cast<float>(i * 10);
    }

    // 2. Host → Device
    d_a.copy_from(h_a);
    d_b.copy_from(h_b);

    // 3. 执行 CUDA 算子
    add_cuda(d_a, d_b, d_c);

    // 4. Device → Host
    h_c.copy_from(d_c);

    // 5. 验证
    for (std::size_t i = 0; i < N; ++i)
    {
        float expected = static_cast<float>(i * 11);
        float actual = h_c.ptr<float>()[i];

        if (std::fabs(actual - expected) > 1e-6f)
        {
            throw std::runtime_error(
                "CUDA Add result mismatch");
        }

        std::cout << actual << " ";
    }

    std::cout << "\nAll tests passed!\n";

    return 0;
}