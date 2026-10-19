#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "minillm/base/data_type.h"
#include "minillm/base/device.h"
#include "minillm/memory/buffer.h"

namespace minillm
{

    class Tensor
    {
    public:
        Tensor(
            std::vector<std::size_t> shape,
            DataType dtype,
            DeviceType device)
            : shape_(std::move(shape)),
              strides_(make_strides(shape_)),
              dtype_(dtype),
              device_(device),
              buffer_(calculate_bytes(shape_, dtype_), device_)
        {
        }

        bool is_contiguous() const noexcept;

        void reshape_(
            std::vector<std::size_t> new_shape);

        std::size_t numel() const
        {
            return calculate_numel(shape_);
        }

        std::size_t bytes() const
        {
            return calculate_bytes(shape_, dtype_);
        }

        std::size_t ndim() const
        {
            return shape_.size();
        }

        const std::vector<std::size_t> &shape() const
        {
            return shape_;
        }

        const std::vector<std::size_t> &strides() const
        {
            return strides_;
        }

        DataType dtype() const
        {
            return dtype_;
        }

        DeviceType device() const
        {
            return device_;
        }

        void *data()
        {
            return buffer_.data();
        }

        const void *data() const
        {
            return buffer_.data();
        }

        template <typename T>
        T *ptr()
        {
            return buffer_.ptr<T>();
        }

        template <typename T>
        const T *ptr() const
        {
            return buffer_.ptr<T>();
        }

        std::size_t offset(
            const std::vector<std::size_t> &indices) const
        {
            if (indices.size() != ndim())
            {
                throw std::invalid_argument(
                    "Tensor index rank mismatch");
            }

            std::size_t result = 0;

            for (std::size_t i = 0; i < ndim(); ++i)
            {
                if (indices[i] >= shape_[i])
                {
                    throw std::out_of_range(
                        "Tensor index out of range");
                }

                result += indices[i] * strides_[i];
            }

            return result;
        }

        // V1: 仅支持 CPU Float32 的直接元素访问
        float &at_float(
            const std::vector<std::size_t> &indices)
        {
            check_cpu_float32();
            return ptr<float>()[offset(indices)];
        }

        const float &at_float(
            const std::vector<std::size_t> &indices) const
        {
            check_cpu_float32();
            return ptr<float>()[offset(indices)];
        }

        void copy_from(const Tensor &src)
        {
            if (this == &src)
            {
                return;
            }

            if (shape_ != src.shape_ ||
                dtype_ != src.dtype_)
            {
                throw std::invalid_argument(
                    "Tensor shape or dtype mismatch");
            }

            if (bytes() == 0)
            {
                return;
            }

            copy_buffer(buffer_, src.buffer_, bytes());
        }

    private:
        void check_cpu_float32() const
        {
            if (device_ != DeviceType::CPU ||
                dtype_ != DataType::FLOAT32)
            {
                throw std::logic_error(
                    "at_float requires CPU Float32 tensor");
            }
        }

        static std::size_t calculate_numel(
            const std::vector<std::size_t> &shape)
        {
            std::size_t count = 1;

            for (std::size_t dim : shape)
            {
                if (dim != 0 &&
                    count > std::numeric_limits<std::size_t>::max() / dim)
                {
                    throw std::overflow_error(
                        "Tensor element count overflow");
                }

                count *= dim;
            }

            return count;
        }

        static std::size_t calculate_bytes(
            const std::vector<std::size_t> &shape,
            DataType dtype)
        {
            const std::size_t element_size =
                data_type_size(dtype);

            if (element_size == 0)
            {
                throw std::invalid_argument(
                    "Unsupported tensor dtype");
            }

            const std::size_t count =
                calculate_numel(shape);

            if (count >
                std::numeric_limits<std::size_t>::max() /
                    element_size)
            {
                throw std::overflow_error(
                    "Tensor byte size overflow");
            }

            return count * element_size;
        }

        static std::vector<std::size_t> make_strides(
            const std::vector<std::size_t> &shape)
        {
            std::vector<std::size_t> strides(shape.size());

            std::size_t stride = 1;

            for (std::size_t i = shape.size(); i > 0; --i)
            {
                strides[i - 1] = stride;
                stride *= shape[i - 1];
            }

            return strides;
        }

    private:
        std::vector<std::size_t> shape_;
        std::vector<std::size_t> strides_;

        DataType dtype_;
        DeviceType device_;

        Buffer buffer_;
    };

} // namespace minillm