#include <cstddef>

#include <cuda_runtime.h>
#include <nvbench/nvbench.cuh>

#include "minillm/ops/rmsnorm.h"

namespace
{

void bench_rmsnorm(nvbench::state &state)
{
    const std::size_t rows =
        static_cast<std::size_t>(state.get_int64("Rows"));
    const std::size_t hidden =
        static_cast<std::size_t>(state.get_int64("Hidden"));

    minillm::Tensor h_x(
        {rows, hidden},
        minillm::DataType::FLOAT32,
        minillm::DeviceType::CPU);

    minillm::Tensor h_w(
        {hidden},
        minillm::DataType::FLOAT32,
        minillm::DeviceType::CPU);

    minillm::Tensor d_x(
        {rows, hidden},
        minillm::DataType::FLOAT32,
        minillm::DeviceType::CUDA);

    minillm::Tensor d_w(
        {hidden},
        minillm::DataType::FLOAT32,
        minillm::DeviceType::CUDA);

    minillm::Tensor d_y(
        {rows, hidden},
        minillm::DataType::FLOAT32,
        minillm::DeviceType::CUDA);

    for (std::size_t i = 0; i < rows * hidden; ++i)
    {
        h_x.ptr<float>()[i] =
            static_cast<float>(static_cast<int>(i % 23) - 11) * 0.1f;
    }

    for (std::size_t i = 0; i < hidden; ++i)
    {
        h_w.ptr<float>()[i] =
            0.5f + 0.01f * static_cast<float>(i % 17);
    }

    d_x.copy_from(h_x);
    d_w.copy_from(h_w);

    // rmsnorm_cuda 当前使用默认 stream，并在函数内部同步。
    state.set_cuda_stream(
        nvbench::make_cuda_stream_view(cudaStream_t{}));

    state.add_element_count(rows * hidden);
    state.add_global_memory_reads<float>(rows * hidden + hidden);
    state.add_global_memory_writes<float>(rows * hidden);

    state.exec(
        nvbench::exec_tag::sync,
        [&](nvbench::launch &)
        {
            minillm::rmsnorm_cuda(d_x, d_w, d_y);
        });
}

} // namespace

NVBENCH_BENCH(bench_rmsnorm)
    .set_name("rmsnorm")
    .add_int64_axis("Rows", {1, 8, 32})
    .add_int64_axis("Hidden", {1024, 2048, 4096});
