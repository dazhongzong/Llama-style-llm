#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>

#include "minillm/tensor/tensor.h"
#include "minillm/ops/rope_config.h"
namespace minillm
{

    void rope_cpu(
        const Tensor &input,
        Tensor &output,
        std::size_t start_pos = 0,
        float theta_base = 10000.0f);

    void rope_cuda_async(
        const Tensor &input,
        Tensor &output,
        std::size_t start_pos,
        float theta_base,
        cudaStream_t stream);

    void rope_cuda(
        const Tensor &input,
        Tensor &output,
        std::size_t start_pos = 0,
        float theta_base = 10000.0f);

    void rope_cuda_async(
        const Tensor &input,
        Tensor &output,
        std::size_t start_pos,
        const RoPEConfig &config,
        cudaStream_t stream);
}