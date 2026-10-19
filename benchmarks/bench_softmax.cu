#include <cstddef>
#include <nvbench/nvbench.cuh>

#include "minillm/ops/softmax.h"

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

    Tensor h_input(
        {M, K}, DataType::FLOAT32, DeviceType::CPU);

    for (std::size_t i = 0; i < M * K; ++i)
    {
        h_input.ptr<float>()[i] =
            static_cast<float>(
                static_cast<int>(i % 11) - 5) *
            0.1f;
    }
    // GPU Tensor

    Tensor d_input(
        {M, K}, DataType::FLOAT32, DeviceType::CUDA);

    Tensor d_output(
        {M, K}, DataType::FLOAT32, DeviceType::CUDA);

    d_input.copy_from(h_input);

    const SoftmaxKernel kernel =
        version == "Block"
            ? SoftmaxKernel::Block
            : SoftmaxKernel::Warp;

    // 记录逻辑处理元素数量
    state.add_element_count(M * K);

    // NVBench 负责重复执行和 GPU 计时
    state.exec([&](nvbench::launch &launch)
               { softmax_cuda_async(
                     d_input,
                     d_output,
                     launch.get_stream(),
                     kernel); });
}

NVBENCH_BENCH(bench_gemv)
    .add_int64_axis("M", {128, 1024, 4096})
    .add_int64_axis("K", {256, 4096})
    .add_string_axis("Version", {"Block", "Warp"});