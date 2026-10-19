
#include "minillm/model/embedding.h"

#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <vector>

namespace minillm
{
    namespace
    {

        __global__ void embedding_lookup_kernel(
            const float *weight,
            float *output,
            std::size_t token_id,
            std::size_t hidden_size)
        {
            const std::size_t d =
                static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

            if (d < hidden_size)
            {
                output[d] =
                    weight[token_id * hidden_size + d];
            }
        }

    } // namespace

    void embedding_lookup_cuda_async(
        const Tensor &weight,
        std::size_t token_id,
        Tensor &output,
        cudaStream_t stream)
    {
        if (weight.ndim() != 2 ||
            output.ndim() != 1 ||
            weight.device() != DeviceType::CUDA ||
            output.device() != DeviceType::CUDA ||
            weight.dtype() != DataType::FLOAT32 ||
            output.dtype() != DataType::FLOAT32 ||
            !weight.is_contiguous() ||
            !output.is_contiguous())
        {
            throw std::invalid_argument(
                "Invalid Embedding Tensor");
        }

        const std::size_t vocab_size = weight.shape()[0];
        const std::size_t H = weight.shape()[1];

        if (token_id >= vocab_size ||
            output.shape()[0] != H)
        {
            throw std::invalid_argument(
                "Embedding token or shape mismatch");
        }

        constexpr unsigned int threads = 256;

        const std::size_t blocks =
            H / threads + (H % threads != 0);

        if (blocks == 0 ||
            blocks > std::numeric_limits<unsigned int>::max())
        {
            throw std::length_error(
                "Invalid Embedding grid");
        }

        embedding_lookup_kernel<<<
            static_cast<unsigned int>(blocks),
            threads,
            0,
            stream>>>(
            weight.ptr<float>(),
            output.ptr<float>(),
            token_id,
            H);

        const cudaError_t err = cudaGetLastError();

        if (err != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(err));
    }

} // namespace minillm
