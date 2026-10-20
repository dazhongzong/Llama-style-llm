#include "minillm/layers/cuda_kv_cache.h"
#include "kernels/cuda_check.h"
#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace minillm
{

    namespace
    {

        std::size_t positive(std::size_t n)
        {
            if (n == 0)
            {
                throw std::invalid_argument(
                    "KVCache dimension must be positive");
            }
            return n;
        }

    }

    CudaKVCache::CudaKVCache(
        std::size_t capacity,
        std::size_t num_kv_heads,
        std::size_t head_dim)
        : capacity_(positive(capacity)),
          num_kv_heads_(positive(num_kv_heads)),
          head_dim_(positive(head_dim)),
          keys_(
              {capacity_, num_kv_heads_, head_dim_},
              DataType::FLOAT32,
              DeviceType::CUDA),
          values_(
              {capacity_, num_kv_heads_, head_dim_},
              DataType::FLOAT32,
              DeviceType::CUDA)
    {
    }

    void CudaKVCache::rewind_to(
        std::size_t position)
    {
        if (position > capacity_)
        {
            throw std::out_of_range(
                "Cannot rewind KV cache beyond capacity");
        }

        next_pos_ = position;
    }

    void CudaKVCache::append_async(
        const Tensor &key,
        const Tensor &value,
        std::size_t position,
        cudaStream_t stream)
    {
        if (position != next_pos_ ||
            position >= capacity_)
        {
            throw std::out_of_range(
                "Invalid KV position");
        }

        const std::vector<std::size_t> expected = {
            num_kv_heads_, head_dim_};

        if (key.shape() != expected ||
            value.shape() != expected ||
            key.device() != DeviceType::CUDA ||
            value.device() != DeviceType::CUDA ||
            key.dtype() != DataType::FLOAT32 ||
            value.dtype() != DataType::FLOAT32)
        {
            throw std::invalid_argument(
                "Invalid KV tensors");
        }

        const std::size_t count =
            num_kv_heads_ * head_dim_;

        const std::size_t offset =
            position * count;

        const std::size_t bytes =
            count * sizeof(float);

        // Key 追加
        CUDA_CHECK(cudaMemcpyAsync(
            keys_.ptr<float>() + offset,
            key.ptr<float>(),
            bytes,
            cudaMemcpyDeviceToDevice,
            stream));

        // Value 追加
        CUDA_CHECK(cudaMemcpyAsync(
            values_.ptr<float>() + offset,
            value.ptr<float>(),
            bytes,
            cudaMemcpyDeviceToDevice,
            stream));

        // 记录已经提交的缓存位置
        ++next_pos_;
    }

    const float *CudaKVCache::key_device_ptr(
        std::size_t position,
        std::size_t kv_head) const
    {
        if (position >= next_pos_ ||
            kv_head >= num_kv_heads_)
        {
            throw std::out_of_range(
                "Invalid Key cache access");
        }

        return keys_.ptr<float>() +
               (position * num_kv_heads_ + kv_head) * head_dim_;
    }

    const float *CudaKVCache::value_device_ptr(
        std::size_t position,
        std::size_t kv_head) const
    {
        if (position >= next_pos_ ||
            kv_head >= num_kv_heads_)
        {
            throw std::out_of_range(
                "Invalid Value cache access");
        }

        return values_.ptr<float>() +
               (position * num_kv_heads_ + kv_head) * head_dim_;
    }

}
