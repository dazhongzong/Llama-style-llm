#pragma once

#include <cuda_runtime_api.h>

#include "minillm/tensor/tensor.h"

namespace minillm
{

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
