#pragma once

#include <cuda_runtime_api.h>

#include "minillm/tensor/tensor.h"

namespace minillm
{
    namespace
    {
        void get_rmsnorm_shape(
            const Tensor &input,
            std::size_t &rows,
            std::size_t &hidden)
        {
            if (input.ndim() == 1)
            {
                rows = 1;
                hidden = input.shape()[0];
            }
            else if (input.ndim() == 2)
            {
                rows = input.shape()[0];
                hidden = input.shape()[1];
            }
            else
            {
                throw std::invalid_argument(
                    "RMSNorm input must have shape [H] or [rows, H]");
            }
        }
    }

    void rmsnorm_cpu(
        const Tensor &input,
        const Tensor &weight,
        Tensor &output,
        float eps = 1e-5f);

    void rmsnorm_cuda(
        const Tensor &input,
        const Tensor &weight,
        Tensor &output,
        float eps = 1e-5f);

    enum class RMSNormVersion
    {
        Shared,
        Warp
    };

    float benchmark_rmsnorm_cuda(
        const Tensor &input,
        const Tensor &weight,
        Tensor &output,
        RMSNormVersion version,
        int warmup = 20,
        int repeats = 200);

    void rmsnorm_cuda_async(
        const Tensor &input,
        const Tensor &gamma,
        Tensor &output,
        float eps,
        cudaStream_t stream);
}
