
#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>
#include "minillm/tensor/tensor.h"

namespace minillm
{

    void embedding_lookup_cuda_async(
        const Tensor &weight,
        std::size_t token_id,
        Tensor &output,
        cudaStream_t stream);

    class Embedding
    {
    public:
        Embedding(
            std::size_t vocab_size,
            std::size_t hidden_size);

        void load_weight(const Tensor &source);

        void forward_async(
            std::size_t token_id,
            Tensor &output,
            cudaStream_t stream) const;

        const Tensor &weight() const noexcept
        {
            return weight_;
        }

        bool loaded() const noexcept
        {
            return loaded_;
        }

    private:
        std::size_t vocab_size_;
        std::size_t hidden_size_;

        Tensor weight_; // [V,H] CUDA FP32
        bool loaded_ = false;
    };

} // namespace minillm
