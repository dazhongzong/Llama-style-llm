#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>

#include "minillm/tensor/tensor.h"

namespace minillm
{

    class CudaKVCache
    {
    public:
        CudaKVCache(
            std::size_t capacity,
            std::size_t num_kv_heads,
            std::size_t head_dim);

        void append_async(
            const Tensor &key,
            const Tensor &value,
            std::size_t position,
            cudaStream_t stream);

        // 返回 GPU 地址，CPU 不得解引用
        const float *key_device_ptr(
            std::size_t position,
            std::size_t kv_head) const;

        const float *value_device_ptr(
            std::size_t position,
            std::size_t kv_head) const;

        std::size_t size() const noexcept
        {
            return next_pos_;
        }

        std::size_t capacity() const noexcept
        {
            return capacity_;
        }

        std::size_t num_kv_heads() const noexcept
        {
            return num_kv_heads_;
        }

        std::size_t head_dim() const noexcept
        {
            return head_dim_;
        }

        const float *keys_device_data() const noexcept
        {
            return keys_.ptr<float>();
        }

        const float *values_device_data() const noexcept
        {
            return values_.ptr<float>();
        }

    private:
        std::size_t capacity_;
        std::size_t num_kv_heads_;
        std::size_t head_dim_;
        std::size_t next_pos_ = 0;

        Tensor keys_;
        Tensor values_;
    };

}