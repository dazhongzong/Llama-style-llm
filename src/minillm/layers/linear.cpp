#include "minillm/layers/linear.h"
#include "minillm/ops/gemv.h"

#include <stdexcept>
#include <vector>

namespace minillm
{

    namespace
    {

        std::size_t require_positive(std::size_t n)
        {
            if (n == 0)
            {
                throw std::invalid_argument(
                    "Linear dimension must be positive");
            }
            return n;
        }

    } // namespace

    Linear::Linear(
        std::size_t in_features,
        std::size_t out_features)
        : in_features_(require_positive(in_features)),
          out_features_(require_positive(out_features)),
          weight_(
              {out_features_, in_features_},
              DataType::FLOAT32,
              DeviceType::CUDA)
    {
    }

    void Linear::load_weight(const Tensor &source)
    {
        if (source.shape() != weight_.shape() ||
            source.dtype() != DataType::FLOAT32)
        {
            throw std::invalid_argument(
                "Linear weight mismatch");
        }

        // 当前 Buffer/Tensor 的 copy_from
        // 负责 CPU↔CUDA 等设备间数据复制
        weight_.copy_from(source);

        weight_loaded_ = true;
    }

    void Linear::validate_shape(
        const Tensor &input,
        const Tensor &output) const
    {
        if (input.ndim() < 1 ||
            output.ndim() != input.ndim())
        {
            throw std::invalid_argument(
                "Linear rank mismatch");
        }

        if (input.shape().back() != in_features_ ||
            output.shape().back() != out_features_)
        {
            throw std::invalid_argument(
                "Linear feature mismatch");
        }

        // 保证 Batch / Seq 等前置维度一致
        for (std::size_t i = 0;
             i + 1 < input.ndim();
             ++i)
        {
            if (input.shape()[i] != output.shape()[i])
            {
                throw std::invalid_argument(
                    "Linear leading dimensions mismatch");
            }
        }
    }

    void Linear::forward_async(
        const Tensor &input,
        Tensor &output,
        cudaStream_t stream) const
    {
        if (!weight_loaded_)
        {
            throw std::runtime_error(
                "Linear weight is not loaded");
        }

        validate_shape(input, output);

        if (input.device() != DeviceType::CUDA ||
            output.device() != DeviceType::CUDA ||
            input.dtype() != DataType::FLOAT32 ||
            output.dtype() != DataType::FLOAT32)
        {
            throw std::invalid_argument(
                "Linear requires CUDA FLOAT32 tensors");
        }

        // 当前只实现了单向量 GEMV
        if (input.ndim() == 1)
        {
            gemv_cuda_async(
                weight_,
                input,
                output,
                GemvKernel::Scalar,
                stream);
        }
        else
        {
            // 后续接入 GEMM
            throw std::runtime_error(
                "Multi-token Linear not implemented yet");
        }
    }

} // namespace minillm