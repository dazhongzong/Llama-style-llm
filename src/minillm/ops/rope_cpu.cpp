#include "minillm/ops/rope.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace minillm
{

    void rope_cpu(
        const Tensor &input,
        Tensor &output,
        std::size_t start_pos,
        float theta_base)
    {
        if (input.device() != DeviceType::CPU ||
            output.device() != DeviceType::CPU)
        {
            throw std::invalid_argument(
                "RoPE CPU requires CPU tensors");
        }

        if (input.dtype() != DataType::FLOAT32 ||
            output.dtype() != DataType::FLOAT32)
        {
            throw std::invalid_argument(
                "RoPE requires FLOAT32");
        }

        if (input.ndim() != 3 ||
            output.shape() != input.shape())
        {
            throw std::invalid_argument(
                "RoPE shape mismatch");
        }

        const std::size_t seq_len = input.shape()[0];
        const std::size_t heads = input.shape()[1];
        const std::size_t head_dim = input.shape()[2];

        if (head_dim == 0 || head_dim % 2 != 0 ||
            !std::isfinite(theta_base) || theta_base <= 1.0f)
        {
            throw std::invalid_argument(
                "Invalid RoPE parameters");
        }

        if (seq_len > 0 &&
            start_pos >
                std::numeric_limits<std::size_t>::max() - (seq_len - 1))
        {
            throw std::overflow_error(
                "RoPE position overflow");
        }

        const float *x = input.ptr<float>();
        float *y = output.ptr<float>();

        for (std::size_t token = 0; token < seq_len; ++token)
        {
            const std::size_t pos = start_pos + token;

            for (std::size_t head = 0; head < heads; ++head)
            {
                const std::size_t base =
                    (token * heads + head) * head_dim;

                for (std::size_t k = 0;
                     k < head_dim / 2;
                     ++k)
                {
                    const double exponent =
                        2.0 * static_cast<double>(k) / static_cast<double>(head_dim);

                    const double angle =
                        static_cast<double>(pos) /
                        std::pow(
                            static_cast<double>(theta_base),
                            exponent);

                    const double c = std::cos(angle);
                    const double s = std::sin(angle);

                    const float x0 = x[base + 2 * k];
                    const float x1 = x[base + 2 * k + 1];

                    y[base + 2 * k] =
                        static_cast<float>(x0 * c - x1 * s);

                    y[base + 2 * k + 1] =
                        static_cast<float>(x0 * s + x1 * c);
                }
            }
        }
    }

} // namespace minillm