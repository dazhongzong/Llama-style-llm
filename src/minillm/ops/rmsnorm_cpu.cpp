#include "minillm/ops/rmsnorm.h"

#include <cmath>
#include <stdexcept>

namespace minillm
{

    void rmsnorm_cpu(
        const Tensor &input,
        const Tensor &weight,
        Tensor &output,
        float eps)
    {
        // 1. 检查 Device
        if (input.device() != DeviceType::CPU ||
            weight.device() != DeviceType::CPU ||
            output.device() != DeviceType::CPU)
        {
            throw std::invalid_argument(
                "rmsnorm_cpu requires CPU tensors");
        }

        // 2. 第一版只支持 FP32
        if (input.dtype() != DataType::FLOAT32 ||
            weight.dtype() != DataType::FLOAT32 ||
            output.dtype() != DataType::FLOAT32)
        {
            throw std::invalid_argument(
                "rmsnorm_cpu requires Float32");
        }

        // 3. 检查维度
        if ((input.ndim() != 1 && input.ndim() != 2) ||
            weight.ndim() != 1 ||
            output.shape() != input.shape())
        {
            throw std::invalid_argument(
                "RMSNorm tensor shape mismatch");
        }

        std::size_t rows = 0;
        std::size_t hidden = 0;
        get_rmsnorm_shape(input, rows, hidden);

        if (hidden == 0 || weight.numel() != hidden ||
            !std::isfinite(eps) || eps <= 0.0f)
        {
            throw std::invalid_argument(
                "Invalid RMSNorm parameters");
        }

        const float *x = input.ptr<float>();
        const float *gamma = weight.ptr<float>();
        float *y = output.ptr<float>();

        // 4. 每一行独立计算
        for (std::size_t row = 0; row < rows; ++row)
        {
            const std::size_t base = row * hidden;

            float sum_sq = 0.0f;

            // 5. 计算平方和
            for (std::size_t col = 0; col < hidden; ++col)
            {
                const float value = x[base + col];

                sum_sq += value * value;
            }

            // 6. 计算归一化系数
            const float mean_sq =
                sum_sq / static_cast<float>(hidden);

            const float inv_rms =
                1.0f / std::sqrt(mean_sq + eps);

            // 7. 归一化并乘权重
            for (std::size_t col = 0; col < hidden; ++col)
            {
                y[base + col] =
                    x[base + col] *
                    inv_rms *
                    gamma[col];
            }
        }
    }

}
