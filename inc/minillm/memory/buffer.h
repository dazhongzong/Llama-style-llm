#pragma once

#include <cstddef>
#include <cstring>
#include <memory>
#include <stdexcept>

#include <cuda_runtime.h>

#include "minillm/base/device.h"

namespace minillm
{

    class Buffer
    {
    public:
        Buffer() = delete;

        Buffer(
            const Buffer &) = delete;

        Buffer &operator=(
            const Buffer &) = delete;

        Buffer(
            Buffer &&other) noexcept
            : data_(other.data_),
              bytes_(other.bytes_),
              device_(other.device_)
        {
            other.data_ = nullptr;
            other.bytes_ = 0;
        }

        Buffer &operator=(
            Buffer &&other) noexcept
        {
            if (this == &other)
            {
                return *this;
            }

            release();

            data_ = other.data_;
            bytes_ = other.bytes_;
            device_ = other.device_;

            other.data_ = nullptr;
            other.bytes_ = 0;

            return *this;
        }

        Buffer(
            std::size_t bytes,
            DeviceType device)
            : bytes_(bytes),
              device_(device)
        {
            if (bytes_ == 0)
            {
                return;
            }

            if (device_ == DeviceType::CPU)
            {
                data_ =
                    std::allocator<std::byte>{}
                        .allocate(bytes_);
            }
            else if (
                device_ == DeviceType::CUDA)
            {
                const cudaError_t err =
                    cudaMalloc(
                        &data_,
                        bytes_);

                if (err != cudaSuccess)
                {
                    throw std::runtime_error(
                        cudaGetErrorString(err));
                }
            }
            else
            {
                throw std::runtime_error(
                    "Unsupported device type");
            }
        }

        ~Buffer() noexcept
        {
            release();
        }

        void *data() noexcept
        {
            return data_;
        }

        const void *data() const noexcept
        {
            return data_;
        }

        std::size_t bytes() const noexcept
        {
            return bytes_;
        }

        DeviceType device() const noexcept
        {
            return device_;
        }

        template <typename T>
        T *ptr() noexcept
        {
            return static_cast<T *>(data_);
        }

        template <typename T>
        const T *ptr() const noexcept
        {
            return static_cast<const T *>(
                data_);
        }

    private:
        void release() noexcept
        {
            if (data_ == nullptr)
            {
                return;
            }

            if (device_ == DeviceType::CPU)
            {
                std::allocator<std::byte>{}
                    .deallocate(
                        static_cast<std::byte *>(
                            data_),
                        bytes_);
            }
            else if (
                device_ == DeviceType::CUDA)
            {
                cudaFree(data_);
            }

            data_ = nullptr;
            bytes_ = 0;
        }

    private:
        void *data_ = nullptr;

        std::size_t bytes_ = 0;

        DeviceType device_ =
            DeviceType::CPU;
    };

    inline void copy_buffer(
        Buffer &dst,
        const Buffer &src,
        std::size_t bytes)
    {
        if (bytes > src.bytes() || bytes > dst.bytes())
        {
            throw std::runtime_error(
                "Copy size exceeds buffer size");
        }

        const DeviceType src_device =
            src.device();

        const DeviceType dst_device =
            dst.device();

        if (src_device == DeviceType::CPU && dst_device == DeviceType::CPU)
        {
            std::memcpy(
                dst.data(),
                src.data(),
                bytes);
        }
        else if (
            src_device == DeviceType::CPU && dst_device == DeviceType::CUDA)
        {
            const cudaError_t err =
                cudaMemcpy(
                    dst.data(),
                    src.data(),
                    bytes,
                    cudaMemcpyHostToDevice);

            if (err != cudaSuccess)
            {
                throw std::runtime_error(
                    cudaGetErrorString(err));
            }
        }
        else if (
            src_device == DeviceType::CUDA && dst_device == DeviceType::CPU)
        {
            const cudaError_t err =
                cudaMemcpy(
                    dst.data(),
                    src.data(),
                    bytes,
                    cudaMemcpyDeviceToHost);

            if (err != cudaSuccess)
            {
                throw std::runtime_error(
                    cudaGetErrorString(err));
            }
        }
        else if (
            src_device == DeviceType::CUDA && dst_device == DeviceType::CUDA)
        {
            const cudaError_t err =
                cudaMemcpy(
                    dst.data(),
                    src.data(),
                    bytes,
                    cudaMemcpyDeviceToDevice);

            if (err != cudaSuccess)
            {
                throw std::runtime_error(
                    cudaGetErrorString(err));
            }
        }
    }

} // namespace minillm