#pragma once
#include <cstddef>
#include <cstdint>
#include <bits/floatn-common.h>
namespace minillm
{
    enum class DataType
    {
        FLOAT32 = 0,
        FLOAT16 = 1,
        INT8 = 2,
        INT32 = 3
    };

    inline std::size_t data_type_size(DataType type)
    {
        switch (type)
        {
        case DataType::FLOAT32:
            return 4;

        case DataType::FLOAT16:
            return 2;

        case DataType::INT8:
            return 1;

        case DataType::INT32:
            return 4;
        }

        return 0;
    }

}