#pragma once

namespace minillm
{

    enum class RoPELayout
    {
        Interleaved,
        HalfSplit
    };

    struct RoPEConfig
    {
        float theta = 10000.0f;
        RoPELayout layout = RoPELayout::Interleaved;
    };

} // namespace minillm