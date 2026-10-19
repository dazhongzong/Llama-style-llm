
#include "minillm/model/embedding.h"

#include <stdexcept>
#include <vector>

namespace minillm
{

    Embedding::Embedding(
        std::size_t vocab_size,
        std::size_t hidden_size)
        : vocab_size_(vocab_size),
          hidden_size_(hidden_size),
          weight_(
              {vocab_size, hidden_size},
              DataType::FLOAT32,
              DeviceType::CUDA)
    {
        if (vocab_size == 0 || hidden_size == 0)
            throw std::invalid_argument(
                "Invalid Embedding dimensions");
    }

    void Embedding::load_weight(const Tensor &source)
    {
        if (source.shape() !=
                std::vector<std::size_t>{
                    vocab_size_, hidden_size_} ||
            source.dtype() != DataType::FLOAT32 ||
            !source.is_contiguous())
        {
            throw std::invalid_argument(
                "Invalid Embedding weight");
        }

        weight_.copy_from(source);
        loaded_ = true;
    }

    void Embedding::forward_async(
        std::size_t token_id,
        Tensor &output,
        cudaStream_t stream) const
    {
        if (!loaded_)
            throw std::runtime_error(
                "Embedding weight not loaded");

        embedding_lookup_cuda_async(
            weight_,
            token_id,
            output,
            stream);
    }

} // namespace minillm
