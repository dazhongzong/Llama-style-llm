#include <cmath>
#include <iostream>
#include <stdexcept>

#include "minillm/layers/attention.h"
#include "minillm/layers/attention_cuda.h"

int main()
{
    using namespace minillm;

    constexpr std::size_t Hq = 4;
    constexpr std::size_t Hkv = 2;
    constexpr std::size_t D = 64;
    constexpr std::size_t Capacity = 8;

    KVCache cpu_cache(Capacity, Hkv, D);
    CudaKVCache gpu_cache(Capacity, Hkv, D);

    AttentionWorkspace workspace(Hq, Capacity);

    Tensor h_q({Hq, D}, DataType::FLOAT32, DeviceType::CPU);
    Tensor h_k({Hkv, D}, DataType::FLOAT32, DeviceType::CPU);
    Tensor h_v({Hkv, D}, DataType::FLOAT32, DeviceType::CPU);

    Tensor h_expected({Hq, D}, DataType::FLOAT32, DeviceType::CPU);
    Tensor h_actual({Hq, D}, DataType::FLOAT32, DeviceType::CPU);

    Tensor d_q({Hq, D}, DataType::FLOAT32, DeviceType::CUDA);
    Tensor d_k({Hkv, D}, DataType::FLOAT32, DeviceType::CUDA);
    Tensor d_v({Hkv, D}, DataType::FLOAT32, DeviceType::CUDA);
    Tensor d_out({Hq, D}, DataType::FLOAT32, DeviceType::CUDA);

    for (std::size_t pos = 0; pos < 3; ++pos)
    {
        // 构造不同 Token 的输入
        for (std::size_t i = 0; i < Hq * D; ++i)
        {
            h_q.ptr<float>()[i] =
                0.02f * static_cast<float>(
                            static_cast<int>((i + pos) % 13) - 6);
        }

        for (std::size_t i = 0; i < Hkv * D; ++i)
        {
            h_k.ptr<float>()[i] =
                0.03f * static_cast<float>(
                            static_cast<int>((i + pos * 3) % 11) - 5);

            h_v.ptr<float>()[i] =
                0.05f * static_cast<float>(
                            static_cast<int>((i + pos * 5) % 17) - 8);
        }

        // CPU Reference：内部追加到 CPU KV Cache
        attention_decode_cpu(
            h_q, h_k, h_v,
            cpu_cache, h_expected, pos);

        // H2D
        d_q.copy_from(h_q);
        d_k.copy_from(h_k);
        d_v.copy_from(h_v);

        // GPU Append
        gpu_cache.append_async(
            d_k, d_v, pos, nullptr);

        // GPU Attention
        attention_decode_cuda_async(
            d_q,
            gpu_cache,
            d_out,
            workspace,
            nullptr);

        // 等待本轮 GPU 工作完成
        const cudaError_t err =
            cudaStreamSynchronize(nullptr);

        if (err != cudaSuccess)
        {
            throw std::runtime_error(
                cudaGetErrorString(err));
        }

        d_out.copy_from(d_out); // 不需要，删除这一行
        h_actual.copy_from(d_out);

        // 与 CPU 比较
        for (std::size_t i = 0; i < Hq * D; ++i)
        {
            const float expected =
                h_expected.ptr<float>()[i];

            const float actual =
                h_actual.ptr<float>()[i];

            const float tolerance =
                1e-4f + 1e-4f * std::fabs(expected);

            if (!std::isfinite(actual) ||
                std::fabs(actual - expected) > tolerance)
            {
                throw std::runtime_error(
                    "CUDA Attention mismatch");
            }
        }

        std::cout
            << "[PASS] Attention Token "
            << pos << '\n';
    }

    std::cout << "All CUDA Attention tests passed!\n";
}