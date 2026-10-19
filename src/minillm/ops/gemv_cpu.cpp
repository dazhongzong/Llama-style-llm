#include "minillm/ops/gemv.h"

#include <stdexcept>

namespace minillm
{

    void gemv_cpu(
        const Tensor &weight,
        const Tensor &input,
        Tensor &output)
    {
        // 1. 检查 Device
        if (weight.device() != DeviceType::CPU ||
            input.device() != DeviceType::CPU ||
            output.device() != DeviceType::CPU)
        {
            throw std::invalid_argument(
                "gemv_cpu requires CPU tensors");
        }

        // 2. 检查 DataType
        if (weight.dtype() != DataType::FLOAT32 ||
            input.dtype() != DataType::FLOAT32 ||
            output.dtype() != DataType::FLOAT32)
        {
            throw std::invalid_argument(
                "gemv_cpu requires FLOAT32");
        }

        // 3. 检查 Tensor 维度
        if (weight.ndim() != 2 ||
            input.ndim() != 1 ||
            output.ndim() != 1)
        {
            throw std::invalid_argument(
                "Invalid GEMV dimensions");
        }

        const std::size_t M = weight.shape()[0];
        const std::size_t K = weight.shape()[1];

        // 4. 检查 Shape
        if (K == 0 ||
            input.shape()[0] != K ||
            output.shape()[0] != M)
        {
            throw std::invalid_argument(
                "GEMV shape mismatch");
        }

        const float *W = weight.ptr<float>();
        const float *x = input.ptr<float>();
        float *y = output.ptr<float>();

        // 5. GEMV
        for (std::size_t row = 0; row < M; ++row)
        {
            float sum = 0.0f;

            for (std::size_t col = 0; col < K; ++col)
            {
                sum +=
                    W[row * K + col] * x[col];
            }

            y[row] = sum;
        }
    }

} // namespace minillm