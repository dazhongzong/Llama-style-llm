#pragma once

#include <cuda_runtime_api.h>
#include "minillm/tensor/tensor.h"

namespace minillm
{

    enum class SoftmaxKernel
    {
        Block,
        Warp
    };

    void softmax_cpu(
        const Tensor &input,
        Tensor &output);

    void softmax_cuda_async(
        const Tensor &input,
        Tensor &output,
        cudaStream_t stream,
        SoftmaxKernel kernel = SoftmaxKernel::Block);

    void softmax_cuda(
        const Tensor &input,
        Tensor &output,
        SoftmaxKernel kernel = SoftmaxKernel::Block);

}
