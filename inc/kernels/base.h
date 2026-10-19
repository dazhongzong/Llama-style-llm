#pragma once
#include <cstddef>

namespace minillm
{
    constexpr std::size_t BLOCK_SIZE = 256;
    constexpr std::size_t WARP_SIZE = 32;
    constexpr std::size_t MAX_THREADS_PER_BLOCK = 1024;

}