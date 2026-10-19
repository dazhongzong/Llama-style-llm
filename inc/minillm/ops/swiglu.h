#pragma once

#include <cuda_runtime_api.h>

#include "minillm/tensor/tensor.h"

namespace minillm
{

    void swiglu_cpu(
        const Tensor &gate,
        const Tensor &up,
        Tensor &output);

    void swiglu_cuda_async(
        const Tensor &gate,
        const Tensor &up,
        Tensor &output,
        cudaStream_t stream);

    void swiglu_cuda(
        const Tensor &gate,
        const Tensor &up,
        Tensor &output);

} // namespace minillm