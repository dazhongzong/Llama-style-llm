#include "minillm/ops/softmax.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace minillm
{

    void softmax_cpu(
        const Tensor &input,
        Tensor &output)
    {
        if (input.device() != DeviceType::CPU ||
            output.device() != DeviceType::CPU)
        {
            throw std::invalid_argument(
                "Softmax requires CPU tensors");
        }

        if (input.dtype() != DataType::FLOAT32 ||
            output.dtype() != DataType::FLOAT32)
        {
            throw std::invalid_argument(
                "Softmax requires FLOAT32 tensors");
        }

        if (input.ndim() != 2 ||
            output.shape() != input.shape())
        {
            throw std::invalid_argument(
                "Softmax shape mismatch");
        }

        const std::size_t rows = input.shape()[0];
        const std::size_t hidden = input.shape()[1];

        if (hidden == 0)
        {
            throw std::invalid_argument(
                "Softmax hidden size must be positive");
        }

        const float *x = input.ptr<float>();
        float *y = output.ptr<float>();

        for (std::size_t row = 0; row < rows; ++row)
        {
            const std::size_t base = row * hidden;

            // 1. 求当前行最大值
            float max_value =
                -std::numeric_limits<float>::infinity();

            for (std::size_t col = 0; col < hidden; ++col)
            {
                max_value = std::max(
                    max_value,
                    x[base + col]);
            }

            // 2. 计算 exp(x - max) 的总和
            double sum = 0.0;

            for (std::size_t col = 0; col < hidden; ++col)
            {
                sum += std::exp(
                    static_cast<double>(x[base + col]) -
                    static_cast<double>(max_value));
            }

            // 3. 归一化
            for (std::size_t col = 0; col < hidden; ++col)
            {
                const double numerator =
                    std::exp(
                        static_cast<double>(x[base + col]) -
                        static_cast<double>(max_value));

                y[base + col] = static_cast<float>(
                    numerator / sum);
            }
        }
    }

}