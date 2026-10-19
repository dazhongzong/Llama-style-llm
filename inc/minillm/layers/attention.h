#pragma once

#include "minillm/layers/kv_cache.h"
#include <stdexcept>
namespace minillm
{

    void attention_decode_cpu(
        const Tensor &q,
        const Tensor &k_new,
        const Tensor &v_new,
        KVCache &cache,
        Tensor &output,
        std::size_t position);

}