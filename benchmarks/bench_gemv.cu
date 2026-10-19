#include <cstddef>
#include <nvbench/nvbench.cuh>

#include "minillm/ops/gemv.h"

using namespace minillm;

void bench_gemv(nvbench::state &state)
{
    const std::size_t M =
        static_cast<std::size_t>(
            state.get_int64("M"));

    const std::size_t K =
        static_cast<std::size_t>(
            state.get_int64("K"));

    const auto version =
        state.get_string("Version");

    // CPU 数据准备
    Tensor h_weight(
        {M, K}, DataType::FLOAT32, DeviceType::CPU);

    Tensor h_input(
        {K}, DataType::FLOAT32, DeviceType::CPU);

    for (std::size_t i = 0; i < M * K; ++i)
    {
        h_weight.ptr<float>()[i] =
            static_cast<float>(
                static_cast<int>(i % 17) - 8) *
            0.01f;
    }

    for (std::size_t i = 0; i < K; ++i)
    {
        h_input.ptr<float>()[i] =
            static_cast<float>(
                static_cast<int>(i % 11) - 5) *
            0.1f;
    }

    // GPU Tensor
    Tensor d_weight(
        {M, K}, DataType::FLOAT32, DeviceType::CUDA);

    Tensor d_input(
        {K}, DataType::FLOAT32, DeviceType::CUDA);

    Tensor d_output(
        {M}, DataType::FLOAT32, DeviceType::CUDA);

    d_weight.copy_from(h_weight);
    d_input.copy_from(h_input);

    const GemvKernel kernel =
        version == "Vec4"
            ? GemvKernel::Vec4
            : GemvKernel::Scalar;

    // 记录逻辑处理元素数量
    state.add_element_count(M * K);

    // NVBench 负责重复执行和 GPU 计时
    state.exec([&](nvbench::launch &launch)
               { gemv_cuda_async(
                     d_weight,
                     d_input,
                     d_output,
                     kernel,
                     launch.get_stream()); });
}

NVBENCH_BENCH(bench_gemv)
    .add_int64_axis("M", {128, 1024, 4096})
    .add_int64_axis("K", {256, 4096, 4097})
    .add_string_axis("Version", {"Scalar", "Vec4"});