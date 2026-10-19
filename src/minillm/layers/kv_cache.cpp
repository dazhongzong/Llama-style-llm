#include "minillm/layers/kv_cache.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace minillm
{

    namespace
    {
        std::size_t positive(std::size_t n)
        {
            if (n == 0)
                throw std::invalid_argument("Zero cache dimension");
            return n;
        }
    }

    KVCache::KVCache(
        std::size_t capacity,
        std::size_t num_kv_heads,
        std::size_t head_dim)
        : capacity_(positive(capacity)),
          num_kv_heads_(positive(num_kv_heads)),
          head_dim_(positive(head_dim)),
          keys_(
              {capacity_, num_kv_heads_, head_dim_},
              DataType::FLOAT32, DeviceType::CPU),
          values_(
              {capacity_, num_kv_heads_, head_dim_},
              DataType::FLOAT32, DeviceType::CPU)
    {
    }

    void KVCache::append(
        const Tensor &key,
        const Tensor &value,
        std::size_t position)
    {
        if (position != next_pos_ || position >= capacity_)
            throw std::out_of_range("Invalid KV position");

        const std::vector<std::size_t> expected = {
            num_kv_heads_, head_dim_};

        if (key.shape() != expected ||
            value.shape() != expected ||
            key.device() != DeviceType::CPU ||
            value.device() != DeviceType::CPU ||
            key.dtype() != DataType::FLOAT32 ||
            value.dtype() != DataType::FLOAT32)
        {
            throw std::invalid_argument("Invalid KV tensor");
        }

        const std::size_t count = num_kv_heads_ * head_dim_;
        const std::size_t offset = position * count;

        std::copy_n(
            key.ptr<float>(), count,
            keys_.ptr<float>() + offset);

        std::copy_n(
            value.ptr<float>(), count,
            values_.ptr<float>() + offset);

        ++next_pos_;
    }

    const float *KVCache::key_at(
        std::size_t position,
        std::size_t kv_head) const
    {
        if (position >= next_pos_ || kv_head >= num_kv_heads_)
            throw std::out_of_range("Invalid KV access");

        return keys_.ptr<float>() +
               (position * num_kv_heads_ + kv_head) * head_dim_;
    }

    const float *KVCache::value_at(
        std::size_t position,
        std::size_t kv_head) const
    {
        if (position >= next_pos_ || kv_head >= num_kv_heads_)
            throw std::out_of_range("Invalid KV access");

        return values_.ptr<float>() +
               (position * num_kv_heads_ + kv_head) * head_dim_;
    }

} // namespace minillm