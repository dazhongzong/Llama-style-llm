#include "minillm/ops/rmsnorm.h"

#include <cuda_runtime.h>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace minillm
{
    constexpr int BLOCK_SIZE = 256;
    constexpr int WARP_SIZE = 32;
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    __device__ __forceinline__ float warp_reduce_sum(float value);
    __global__ void rmsnorm_kernel(
        const float *x,
        const float *gamma,
        float *y,
        std::size_t hidden,
        float eps)
    {
        const int tid = threadIdx.x;

        // 当前 Thread 位于 Warp 内的哪个 Lane
        const int lane = tid % WARP_SIZE;

        // 当前 Thread 属于 Block 内第几个 Warp
        const int warp_id = tid / WARP_SIZE;

        const std::size_t row = blockIdx.x;
        const std::size_t base = row * hidden;

        __shared__ float warp_sums[NUM_WARPS];
        __shared__ float shared_inv_rms;

        // 1. 每个 Thread 独立计算局部平方和
        float local_sum = 0.0f;

        for (std::size_t col = tid;
             col < hidden;
             col += BLOCK_SIZE)
        {
            const float v = x[base + col];
            local_sum += v * v;
        }

        // 2. Warp 内归约
        local_sum = warp_reduce_sum(local_sum);

        // 3. 每个 Warp 的 Lane 0 写入一个结果
        if (lane == 0)
        {
            warp_sums[warp_id] = local_sum;
        }

        // 必须等全部 Warp 的结果写入
        __syncthreads();

        // 4. 由 Warp 0 负责第二次归约
        if (warp_id == 0)
        {
            float value =
                (lane < NUM_WARPS)
                    ? warp_sums[lane]
                    : 0.0f;

            value = warp_reduce_sum(value);

            // Warp 0 的 Lane 0 得到整行平方和
            if (lane == 0)
            {
                const float mean_sq =
                    value / static_cast<float>(hidden);

                shared_inv_rms =
                    1.0f / sqrtf(mean_sq + eps);
            }
        }

        // 等待归一化系数计算完成
        __syncthreads();

        // 5. 每个 Thread 独立计算输出
        for (std::size_t col = tid;
             col < hidden;
             col += BLOCK_SIZE)
        {
            y[base + col] =
                x[base + col] *
                shared_inv_rms *
                gamma[col];
        }
    }
    void rmsnorm_cuda_async(
        const Tensor &input,
        const Tensor &weight,
        Tensor &output,
        float eps,
        cudaStream_t stream)
    {
        if (input.device() != DeviceType::CUDA ||
            weight.device() != DeviceType::CUDA ||
            output.device() != DeviceType::CUDA)
        {
            throw std::invalid_argument(
                "rmsnorm_cuda_async requires CUDA tensors");
        }

        if (input.dtype() != DataType::FLOAT32 ||
            weight.dtype() != DataType::FLOAT32 ||
            output.dtype() != DataType::FLOAT32)
        {
            throw std::invalid_argument(
                "rmsnorm_cuda_async requires FLOAT32 tensors");
        }

        if ((input.ndim() != 1 && input.ndim() != 2) ||
            weight.ndim() != 1 ||
            output.shape() != input.shape())
        {
            throw std::invalid_argument(
                "RMSNorm shape mismatch");
        }

        std::size_t rows = 0;
        std::size_t hidden = 0;
        get_rmsnorm_shape(input, rows, hidden);

        if (hidden == 0 ||
            weight.numel() != hidden ||
            !std::isfinite(eps) ||
            eps <= 0.0f)
        {
            throw std::invalid_argument(
                "Invalid RMSNorm parameters");
        }

        // 避免启动空 Grid
        if (rows == 0)
        {
            return;
        }

        // 这里限制为常见 CUDA 一维 Grid 的安全范围
        if (rows >
            static_cast<std::size_t>(
                std::numeric_limits<int>::max()))
        {
            throw std::length_error(
                "Too many RMSNorm rows");
        }

        rmsnorm_kernel<<<static_cast<unsigned int>(rows),
                         BLOCK_SIZE,
                         0,
                         stream>>>(
            input.ptr<float>(),
            weight.ptr<float>(),
            output.ptr<float>(),
            hidden,
            eps);

        cudaError_t err = cudaGetLastError();

        if (err != cudaSuccess)
        {
            throw std::runtime_error(
                std::string("RMSNorm launch error: ") +
                cudaGetErrorString(err));
        }
    }

    // Host 端同步算子接口
    void rmsnorm_cuda(
        const Tensor &input,
        const Tensor &weight,
        Tensor &output,
        float eps)
    {
        rmsnorm_cuda_async(
            input,
            weight,
            output,
            eps,
            nullptr);

        const cudaError_t err =
            cudaStreamSynchronize(nullptr);

        if (err != cudaSuccess)
        {
            throw std::runtime_error(
                std::string("RMSNorm execution error: ") +
                cudaGetErrorString(err));
        }
    }

    __device__ __forceinline__ float warp_reduce_sum(float value)
    {
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        {
            value += __shfl_down_sync(0xffffffff, value, offset);
        }
        return value;
    }

    __global__ void rmsnorm_warp_kernel(
        const float *x,
        const float *gamma,
        float *y,
        std::size_t hidden,
        float eps)
    {
        const int tid = threadIdx.x;

        // 当前 Thread 位于 Warp 内的哪个 Lane
        const int lane = tid % WARP_SIZE;

        // 当前 Thread 属于 Block 内第几个 Warp
        const int warp_id = tid / WARP_SIZE;

        const std::size_t row = blockIdx.x;
        const std::size_t base = row * hidden;

        __shared__ float warp_sums[NUM_WARPS];
        __shared__ float shared_inv_rms;

        // 1. 每个 Thread 独立计算局部平方和
        float local_sum = 0.0f;

        for (std::size_t col = tid;
             col < hidden;
             col += BLOCK_SIZE)
        {
            const float v = x[base + col];
            local_sum += v * v;
        }

        // 2. Warp 内归约
        local_sum = warp_reduce_sum(local_sum);

        // 3. 每个 Warp 的 Lane 0 写入一个结果
        if (lane == 0)
        {
            warp_sums[warp_id] = local_sum;
        }

        // 必须等全部 Warp 的结果写入
        __syncthreads();

        // 4. 由 Warp 0 负责第二次归约
        if (warp_id == 0)
        {
            float value =
                (lane < NUM_WARPS)
                    ? warp_sums[lane]
                    : 0.0f;

            value = warp_reduce_sum(value);

            // Warp 0 的 Lane 0 得到整行平方和
            if (lane == 0)
            {
                const float mean_sq =
                    value / static_cast<float>(hidden);

                shared_inv_rms =
                    1.0f / sqrtf(mean_sq + eps);
            }
        }

        // 等待归一化系数计算完成
        __syncthreads();

        // 5. 每个 Thread 独立计算输出
        for (std::size_t col = tid;
             col < hidden;
             col += BLOCK_SIZE)
        {
            y[base + col] =
                x[base + col] *
                shared_inv_rms *
                gamma[col];
        }
    }

    __global__ void rmsnorm_shared_kernel(
        const float *x,
        const float *gamma,
        float *y,
        std::size_t hidden,
        float eps)
    {
        const int tid = threadIdx.x;
        const std::size_t row = blockIdx.x;
        const std::size_t base = row * hidden;

        __shared__ float shared_sum[BLOCK_SIZE];
        __shared__ float shared_inv_rms;

        // 第一步：每个 Thread 计算局部平方和
        float local_sum = 0.0f;

        for (std::size_t col = tid;
             col < hidden;
             col += blockDim.x)
        {
            const float v = x[base + col];

            local_sum += v * v;
        }

        // 第二步：写入 Shared Memory
        shared_sum[tid] = local_sum;

        __syncthreads();

        // 第三步：Block Reduction
        for (int stride = blockDim.x / 2;
             stride > 0;
             stride /= 2)
        {
            if (tid < stride)
            {
                shared_sum[tid] +=
                    shared_sum[tid + stride];
            }

            __syncthreads();
        }

        // 第四步：计算归一化系数
        if (tid == 0)
        {
            const float mean_sq =
                shared_sum[0] /
                static_cast<float>(hidden);

            shared_inv_rms =
                1.0f / sqrtf(mean_sq + eps);
        }

        __syncthreads();

        // 第五步：各线程独立计算输出
        for (std::size_t col = tid;
             col < hidden;
             col += blockDim.x)
        {
            y[base + col] =
                x[base + col] *
                shared_inv_rms *
                gamma[col];
        }
    }
    // CUDA 错误检查
    void check_cuda(cudaError_t err)
    {
        if (err != cudaSuccess)
        {
            throw std::runtime_error(
                cudaGetErrorString(err));
        }
    }

    // RAII 管理 CUDA Event
    class CudaEvent
    {
    public:
        CudaEvent()
        {
            check_cuda(cudaEventCreate(&event_));
        }

        ~CudaEvent() noexcept
        {
            if (event_)
            {
                cudaEventDestroy(event_);
            }
        }

        CudaEvent(const CudaEvent &) = delete;
        CudaEvent &operator=(const CudaEvent &) = delete;

        cudaEvent_t get() const
        {
            return event_;
        }

    private:
        cudaEvent_t event_{};
    };

    float benchmark_rmsnorm_cuda(
        const Tensor &input,
        const Tensor &weight,
        Tensor &output,
        RMSNormVersion version,
        int warmup,
        int repeats)
    {
        if (warmup < 0 || repeats <= 0)
        {
            throw std::invalid_argument(
                "Invalid benchmark iterations");
        }

        if (input.device() != DeviceType::CUDA ||
            weight.device() != DeviceType::CUDA ||
            output.device() != DeviceType::CUDA ||
            input.dtype() != DataType::FLOAT32 ||
            weight.dtype() != DataType::FLOAT32 ||
            output.dtype() != DataType::FLOAT32 ||
            (input.ndim() != 1 && input.ndim() != 2) ||
            weight.ndim() != 1 ||
            output.shape() != input.shape())
        {
            throw std::invalid_argument(
                "Invalid RMSNorm benchmark tensors");
        }

        std::size_t rows = 0;
        std::size_t hidden = 0;
        get_rmsnorm_shape(input, rows, hidden);

        if (rows == 0 || hidden == 0 ||
            rows > static_cast<std::size_t>(
                       std::numeric_limits<int>::max()) ||
            weight.numel() != hidden)
        {
            throw std::invalid_argument(
                "Invalid RMSNorm benchmark shape");
        }

        constexpr float eps = 1e-5f;

        // 两个版本使用相同的启动配置
        auto launch = [&]()
        {
            if (version == RMSNormVersion::Shared)
            {
                rmsnorm_shared_kernel<<<
                    static_cast<unsigned int>(rows),
                    BLOCK_SIZE>>>(
                    input.ptr<float>(),
                    weight.ptr<float>(),
                    output.ptr<float>(),
                    hidden,
                    eps);
            }
            else
            {
                rmsnorm_warp_kernel<<<
                    static_cast<unsigned int>(rows),
                    BLOCK_SIZE>>>(
                    input.ptr<float>(),
                    weight.ptr<float>(),
                    output.ptr<float>(),
                    hidden,
                    eps);
            }
        };

        // Warmup
        for (int i = 0; i < warmup; ++i)
        {
            launch();
        }

        check_cuda(cudaGetLastError());
        check_cuda(cudaDeviceSynchronize());

        CudaEvent start;
        CudaEvent stop;

        check_cuda(cudaEventRecord(start.get()));

        // 正式计时
        for (int i = 0; i < repeats; ++i)
        {
            launch();
        }

        check_cuda(cudaGetLastError());

        check_cuda(cudaEventRecord(stop.get()));
        check_cuda(cudaEventSynchronize(stop.get()));

        float total_ms = 0.0f;

        check_cuda(cudaEventElapsedTime(
            &total_ms,
            start.get(),
            stop.get()));

        // 返回单次平均耗时（微秒）
        return total_ms * 1000.0f / repeats;
    }

} // namespace minillm
