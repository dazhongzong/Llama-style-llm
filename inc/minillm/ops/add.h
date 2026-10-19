#pragma once

#include "minillm/tensor/tensor.h"

namespace minillm
{

    void add_cuda(
        const Tensor &a,
        const Tensor &b,
        Tensor &out);

    void add_cuda_async(
        const Tensor &a,
        const Tensor &b,
        Tensor &out,
        cudaStream_t stream);

}