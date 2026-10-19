#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <exception>
#include <stdexcept>
#define CUDA_CHECK(call)                                                 \
    do                                                                   \
    {                                                                    \
        cudaError_t err = (call);                                        \
        if (err != cudaSuccess)                                          \
        {                                                                \
            throw std::runtime_error(                                    \
                std::string("CUDA error at ") + __FILE__ + ":" +         \
                std::to_string(__LINE__) + " - " +                       \
                cudaGetErrorName(err) + ": " + cudaGetErrorString(err)); \
        }                                                                \
    } while (0)
