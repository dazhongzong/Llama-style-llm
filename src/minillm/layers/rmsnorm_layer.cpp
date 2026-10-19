#include "minillm/layers/rmsnorm_layer.h"
#include "minillm/ops/rmsnorm.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace minillm
{

    namespace
    {

        std::size_t positive(std::size_t value)
        {
            if (value == 0)
                throw std::invalid_argument("Invalid hidden size");

            return value;
        }

        float valid_eps(float eps)
        {
            if (!std::isfinite(eps) || eps <= 0.0f)
                throw std::invalid_argument("Invalid RMSNorm eps");

            return eps;
        }

    }

    RMSNormLayer::RMSNormLayer(
        std::size_t hidden_size,
        float eps)
        : hidden_size_(positive(hidden_size)),
          eps_(valid_eps(eps)),
          weight_(
              {hidden_size_},
              DataType::FLOAT32,
              DeviceType::CUDA)
    {
    }

    void RMSNormLayer::load_weight(const Tensor &source)
    {
        if (source.shape() !=
                std::vector<std::size_t>{hidden_size_} ||
            source.dtype() != DataType::FLOAT32 ||
            !source.is_contiguous())
        {
            throw std::invalid_argument(
                "Invalid RMSNorm gamma");
        }

        weight_.copy_from(source);
        weight_loaded_ = true;
    }

    void RMSNormLayer::forward_async(
        const Tensor &input,
        Tensor &output,
        cudaStream_t stream) const
    {
        if (!weight_loaded_)
            throw std::runtime_error(
                "RMSNorm weight not loaded");

        const std::vector<std::size_t> expected{
            hidden_size_};

        if (input.shape() != expected ||
            output.shape() != expected)
        {
            throw std::invalid_argument(
                "RMSNormLayer expects [hidden_size]");
        }

        rmsnorm_cuda_async(
            input,
            weight_,
            output,
            eps_,
            stream);
    }

} // namespace minillm