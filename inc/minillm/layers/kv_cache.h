#pragma once
#include "minillm/tensor/tensor.h"
#include <stdexcept>
namespace minillm
{
    class KVCache
    {
    public:
        KVCache(
            std::size_t capacity,
            std::size_t num_kv_heads,
            std::size_t head_dim);

        void append(
            const Tensor &key,
            const Tensor &value,
            std::size_t position);

        const float *key_at(
            std::size_t position,
            std::size_t kv_head) const;

        const float *value_at(
            std::size_t position,
            std::size_t kv_head) const;

        std::size_t size() const noexcept
        {
            return next_pos_;
        }

        std::size_t num_kv_heads() const noexcept
        {
            return num_kv_heads_;
        }

        std::size_t head_dim() const noexcept
        {
            return head_dim_;
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