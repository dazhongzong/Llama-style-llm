#pragma once
#include <cuda_runtime_api.h>
#include "minillm/tensor/tensor.h"

namespace minillm
{

    void gemv_cpu(
        const Tensor &weight,
        const Tensor &input,
        Tensor &output);

    void gemv_cuda(
        const Tensor &weight,
        const Tensor &input,
        Tensor &output);
    enum class GemvKernel
    {
        Scalar,
        Vec4
    };

    // 只提交 GPU 工作，不主动等待完成
    void gemv_cuda_async(
        const Tensor &weight,
        const Tensor &input,
        Tensor &output,
        GemvKernel version,
        cudaStream_t stream);
}