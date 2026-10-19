#include "minillm/ops/swiglu.h"

#include <cmath>
#include <stdexcept>

namespace minillm
{

    void swiglu_cpu(
        const Tensor &gate,
        const Tensor &up,
        Tensor &output)
    {
        if (gate.device() != DeviceType::CPU ||
            up.device() != DeviceType::CPU ||
            output.device() != DeviceType::CPU)
        {
            throw std::invalid_argument(
                "SwiGLU requires CPU tensors");
        }

        if (gate.dtype() != DataType::FLOAT32 ||
            up.dtype() != DataType::FLOAT32 ||
            output.dtype() != DataType::FLOAT32)
        {
            throw std::invalid_argument(
                "SwiGLU requires FLOAT32 tensors");
        }

        if (gate.shape() != up.shape() ||
            gate.shape() != output.shape())
        {
            throw std::invalid_argument(
                "SwiGLU shape mismatch");
        }

        const float *g = gate.ptr<float>();
        const float *u = up.ptr<float>();
        float *y = output.ptr<float>();

        const std::size_t n = gate.numel();

        for (std::size_t i = 0; i < n; ++i)
        {
            const float value = g[i];

            const float silu =
                value / (1.0f + std::exp(-value));

            y[i] = silu * u[i];
        }
    }

} // namespace minillm