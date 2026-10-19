#include "minillm/layers/feed_forward.h"
#include "minillm/ops/swiglu.h"

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <vector>

namespace minillm
{

    FeedForwardWorkspace::FeedForwardWorkspace(
        std::size_t intermediate_size)
        : gate(
              {intermediate_size},
              DataType::FLOAT32,
              DeviceType::CUDA),
          up(
              {intermediate_size},
              DataType::FLOAT32,
              DeviceType::CUDA),
          activated(
              {intermediate_size},
              DataType::FLOAT32,
              DeviceType::CUDA)
    {
    }

    FeedForward::FeedForward(
        std::size_t hidden_size,
        std::size_t intermediate_size)
        : hidden_size_(hidden_size),
          intermediate_size_(intermediate_size),
          gate_proj_(hidden_size, intermediate_size),
          up_proj_(hidden_size, intermediate_size),
          down_proj_(intermediate_size, hidden_size)
    {
    }

    void FeedForward::forward_async(
        const Tensor &input,
        Tensor &output,
        FeedForwardWorkspace &workspace,
        cudaStream_t stream) const
    {
        // 当前 FFN 只支持单 Token Decode
        if (input.shape() !=
                std::vector<std::size_t>{hidden_size_} ||
            output.shape() !=
                std::vector<std::size_t>{hidden_size_})
        {
            throw std::invalid_argument(
                "FeedForward input/output mismatch");
        }

        const std::vector<std::size_t> intermediate_shape = {
            intermediate_size_};

        if (workspace.gate.shape() != intermediate_shape ||
            workspace.up.shape() != intermediate_shape ||
            workspace.activated.shape() != intermediate_shape)
        {
            throw std::invalid_argument(
                "FeedForward workspace mismatch");
        }

        // 1. Gate Projection
        gate_proj_.forward_async(
            input,
            workspace.gate,
            stream);

        // 2. Up Projection
        up_proj_.forward_async(
            input,
            workspace.up,
            stream);

        // 3. SwiGLU
        swiglu_cuda_async(
            workspace.gate,
            workspace.up,
            workspace.activated,
            stream);

        // 4. Down Projection
        down_proj_.forward_async(
            workspace.activated,
            output,
            stream);
    }

    void FeedForward::forward(
        const Tensor &input,
        Tensor &output,
        FeedForwardWorkspace &workspace) const
    {
        forward_async(
            input,
            output,
            workspace,
            nullptr);

        const cudaError_t err =
            cudaStreamSynchronize(nullptr);

        if (err != cudaSuccess)
        {
            throw std::runtime_error(
                cudaGetErrorString(err));
        }
    }

} // namespace minillm