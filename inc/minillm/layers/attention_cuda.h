#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>

#include "minillm/layers/cuda_kv_cache.h"
#include "minillm/tensor/tensor.h"

namespace minillm
{

    class AttentionWorkspace
    {
    public:
        AttentionWorkspace(
            std::size_t num_q_heads,
            std::size_t capacity);

        // 预分配的 GPU 中间结果
        Tensor scores;
        Tensor weights;
    };

    void attention_decode_cuda_async(
        const Tensor &q,
        const CudaKVCache &cache,
        Tensor &output,
        AttentionWorkspace &workspace,
        cudaStream_t stream);

    void attention_decode_cuda(
        const Tensor &q,
        const CudaKVCache &cache,
        Tensor &output,
        AttentionWorkspace &workspace);

}