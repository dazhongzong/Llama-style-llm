#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>

#include "minillm/layers/linear.h"

namespace minillm
{

    class FeedForwardWorkspace
    {
    public:
        explicit FeedForwardWorkspace(
            std::size_t intermediate_size);

        Tensor gate;
        Tensor up;
        Tensor activated;
    };

    class FeedForward
    {
    public:
        FeedForward(
            std::size_t hidden_size,
            std::size_t intermediate_size);

        Linear &gate_proj() noexcept
        {
            return gate_proj_;
        }

        Linear &up_proj() noexcept
        {
            return up_proj_;
        }

        Linear &down_proj() noexcept
        {
            return down_proj_;
        }

        void forward_async(
            const Tensor &input,
            Tensor &output,
            FeedForwardWorkspace &workspace,
            cudaStream_t stream) const;

        void forward(
            const Tensor &input,
            Tensor &output,
            FeedForwardWorkspace &workspace) const;

    private:
        std::size_t hidden_size_;
        std::size_t intermediate_size_;

        Linear gate_proj_;
        Linear up_proj_;
        Linear down_proj_;
    };

} // namespace minillm