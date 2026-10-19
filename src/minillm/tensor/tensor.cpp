#include "minillm/tensor/tensor.h"

namespace minillm
{

    bool Tensor::is_contiguous() const noexcept
    {
        if (shape_.size() != strides_.size())
        {
            return false;
        }

        std::size_t expected_stride = 1;

        // 从最后一维往前检查 Row-Major Stride
        for (std::size_t i = shape_.size(); i-- > 0;)
        {
            if (strides_[i] != expected_stride)
            {
                return false;
            }

            if (shape_[i] == 0 ||
                expected_stride >
                    std::numeric_limits<std::size_t>::max() / shape_[i])
            {
                return false;
            }

            expected_stride *= shape_[i];
        }

        return true;
    }

    void Tensor::reshape_(
        std::vector<std::size_t> new_shape)
    {
        if (!is_contiguous())
        {
            throw std::runtime_error(
                "reshape_ requires contiguous Tensor");
        }

        if (new_shape.empty())
        {
            throw std::invalid_argument(
                "reshape_ requires non-empty shape");
        }

        // 第一步：检查元素数量
        std::size_t new_numel = 1;

        for (std::size_t dim : new_shape)
        {
            if (dim == 0 ||
                new_numel >
                    std::numeric_limits<std::size_t>::max() / dim)
            {
                throw std::invalid_argument(
                    "Invalid reshape dimensions");
            }

            new_numel *= dim;
        }

        if (new_numel != numel())
        {
            throw std::invalid_argument(
                "reshape_ changes element count");
        }

        // 第二步：重新计算 Row-Major Strides
        std::vector<std::size_t> new_strides(
            new_shape.size());

        std::size_t stride = 1;

        for (std::size_t i = new_shape.size(); i-- > 0;)
        {
            new_strides[i] = stride;
            stride *= new_shape[i];
        }

        // 第三步：交换元数据，不改变 Buffer
        shape_.swap(new_shape);
        strides_.swap(new_strides);
    }

} // namespace minillm