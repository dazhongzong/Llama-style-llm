#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>

#include "minillm/tensor/tensor.h"

namespace minillm
{

    class Linear
    {
    public:
        Linear(
            std::size_t in_features,
            std::size_t out_features);

        void load_weight(const Tensor &source);

        void forward_async(
            const Tensor &input,
            Tensor &output,
            cudaStream_t stream) const;

        std::size_t in_features() const noexcept
        {
            return in_features_;
        }

        std::size_t out_features() const noexcept
        {
            return out_features_;
        }

    private:
        void validate_shape(
            const Tensor &input,
            const Tensor &output) const;

        std::size_t in_features_;
        std::size_t out_features_;

        Tensor weight_;
        bool weight_loaded_ = false;
    };

} // namespace minillm