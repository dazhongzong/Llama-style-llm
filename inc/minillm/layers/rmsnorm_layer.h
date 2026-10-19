#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>

#include "minillm/tensor/tensor.h"

namespace minillm
{

    class RMSNormLayer
    {
    public:
        RMSNormLayer(
            std::size_t hidden_size,
            float eps = 1e-5f);

        void load_weight(const Tensor &source);

        void forward_async(
            const Tensor &input,
            Tensor &output,
            cudaStream_t stream) const;

    private:
        std::size_t hidden_size_;
        float eps_;
        Tensor weight_; // gamma [H]
        bool weight_loaded_ = false;
    };

} // namespace minillm