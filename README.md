# CudaLearning

这是一个 CUDA/C++ 学习项目。项目使用 CMake 构建，CUDA 算子编译为静态库，C++ 主程序和单元测试链接该静态库。

## 目录结构

```text
inc/                 C++ 头文件
inc/kernels/         CUDA 头文件
src/                 C++ 源文件，例如 main.cpp、ops/*.cpp
src/kernels/         CUDA 源文件
tests/               单元测试，每个源文件生成一个测试程序
benchmarks/          NVBench benchmark 源文件
configs/             模型注册和运行配置
third_party/         Git 子模块依赖
```

## 环境要求

- CMake >= 3.19
- C++ 编译器，支持 C++20
- CUDA Toolkit，当前配置使用 CUDA 17 语言标准
- Rust 工具链（`cargo`），用于编译 `tokenizers-cpp` 的 Hugging Face tokenizer 后端
- 可用的 NVIDIA 驱动和 CUDA 设备（运行 CUDA 测试时需要）

Python 参考脚本使用 `uv` 管理依赖：

- Python >= 3.10
- uv
- NumPy
- PyTorch
- Transformers

检查工具：

```bash
cmake --version
nvcc --version
g++ --version
uv --version
```

如果尚未安装 Rust，可以使用 rustup 安装 stable 工具链：

```bash
rustup toolchain install stable --profile minimal
rustup default stable
```

## Python 依赖

初始化虚拟环境并安装锁定依赖：

```bash
uv sync
```

使用项目环境运行模型检查脚本：

```bash
uv run python tools/inspect_model.py
```

导出 Transformers 参考 logits：

```bash
uv run python tests/reference/export_logits.py
```

## 文本生成

`app` 通过 [tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp) 读取模型的
`tokenizer.json`，执行 chat template、prompt prefill，
然后使用 KV Cache 进行 greedy 自回归生成：

```bash
cmake --build build --parallel
./build/app \
    --prompt "Explain CUDA kernels in one sentence." \
    --max-new-tokens 64
```

也可以使用 `--prompt -` 从标准输入读取一行文本，适合脚本或管道调用：

```bash
echo "Explain CUDA kernels in one sentence." | \
    ./build/app --prompt - --max-new-tokens 64
```

直接在终端输入一行 prompt 后按回车即可开始生成：

```bash
./build/app --prompt - --max-new-tokens 64
```

如果不希望使用 SmolLM 的 chat template，可以使用原始 prompt：

```bash
./build/app --raw --prompt "Once upon a time" --max-new-tokens 32
```

使用 `--chat` 可以进入持续对话模式。模型权重只加载一次，每轮会保留
此前的 user/assistant 历史并重新执行 prompt prefill：

```bash
./build/app --chat --max-new-tokens 64
```

交互命令：`/help` 查看帮助，`/reset` 清空历史，`/exit` 或 `/quit` 退出。
历史长度超过模型上下文时，使用 `/reset` 开始新的会话。

生成需要可用且兼容的 NVIDIA 驱动。`configs/models.json` 中的模型目录必须
同时包含 `config.json`、`model.safetensors` 和 `tokenizer.json`。

首次拉取项目或更新依赖时，需要初始化 tokenizers-cpp 及其子模块：

```bash
git submodule update --init --recursive
```

添加或升级依赖：

```bash
uv add numpy
uv add torch
uv add transformers
uv lock --upgrade
uv sync
```

`uv.lock` 应提交到 Git，用于保证不同环境的依赖版本一致。

## 模型 Decode Benchmark

benchmark 默认使用已有的 SmolLM2-135M-Instruct。它测量单个增量 decode
步骤，支持通过 `MINILLM_MODEL_DIR` 切换到 TinyLlama 等兼容的 LLaMA 模型。

如果需要 TinyLlama 1.1B，先下载模型文件：

```bash
hf download TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    config.json tokenizer.json tokenizer_config.json model.safetensors \
    --local-dir models/TinyLlama-1.1B-Chat-v1.0
```

构建独立 benchmark 工程：

```bash
cmake -S benchmarks -B build-benchmarks \
    -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build-benchmarks \
    --target bench_transformer_decode --parallel
```

运行 SmolLM2 单 token decode benchmark：

```bash
./build-benchmarks/bench_transformer_decode
```

切换到 TinyLlama：

```bash
MINILLM_MODEL_DIR=models/TinyLlama-1.1B-Chat-v1.0 \
    ./build-benchmarks/bench_transformer_decode
```

该 benchmark 会预填充 1、32、128、256 个 token，然后测量一个增量 decode
步骤的 GPU 时间；模型加载和 prefill 不计入测量区间。

已有 SmolLM2-135M-Instruct 在 NVIDIA GeForce RTX 5070 上的结果记录在
[benchmarks/results/smollm2-135m-rtx5070.md](benchmarks/results/smollm2-135m-rtx5070.md)。
TinyLlama-1.1B-Chat 的结果记录在
[benchmarks/results/tinyllama-1.1b-rtx5070.md](benchmarks/results/tinyllama-1.1b-rtx5070.md)。

## 配置项目

在项目根目录执行：

```bash
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=80
```

`80` 需要根据显卡计算能力调整。例如项目原配置同时构建 `80` 和 `89`：

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES="80;89"
```

如果只想关闭单元测试：

```bash
cmake -S . -B build -DBUILD_TESTING=OFF
```

## 编译

```bash
cmake --build build --parallel
```

编译完成后，主程序位于：

```text
build/app
```

运行主程序：

```bash
./build/app
```

## 运行单元测试

运行所有测试，并在失败时显示详细输出：

```bash
ctest --test-dir build --output-on-failure
```

只运行 `test_core`：

```bash
ctest --test-dir build -R test_core --output-on-failure
```

也可以直接运行测试可执行文件：

```bash
./build/test_test_core
```

当前的 `tests/test_core.cpp` 会测试：

- Tensor 的维度、元素数量、字节数和 stride；
- CPU 与 CUDA 之间的数据拷贝；
- CUDA 向量加法算子。

## 使用 NVBench

benchmark 是独立的 CMake 工程，不参与根工程的 `app` 和单元测试构建。
当前 NVBench 主分支要求 CMake 3.30.4 或更高版本，配置时会从 NVIDIA
官方仓库获取依赖。

在项目根目录配置并编译所有 benchmark：

```bash
cmake -S benchmarks -B build-benchmarks \
  -DCMAKE_CUDA_ARCHITECTURES=80

# 如果该构建目录以前配置过失效的本地依赖路径，可先清除缓存：
# cmake -S benchmarks -B build-benchmarks \
#   -U 'FETCHCONTENT_SOURCE_DIR_*' \
#   -DCMAKE_CUDA_ARCHITECTURES=80

cmake --build build-benchmarks \
  --target all_benchmarks \
  --parallel
```

只编译某一个 benchmark：

```bash
cmake --build build-benchmarks \
  --target bench_rmsnorm_nvbench \
  --parallel
```

运行 benchmark：

```bash
./build-benchmarks/bench_rmsnorm_nvbench
./build-benchmarks/bench_rmsnorm_nvbench --help
```

根工程仍然单独配置普通程序和单元测试：

```bash
cmake -S . -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

## 新增测试

在 `tests/` 下新增一个带有 `main()` 的测试源文件，例如：

```text
tests/test_tensor.cpp
```

重新配置并编译：

```bash
cmake -S . -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

CMake 会自动生成对应的测试目标：

```text
build/test_test_tensor
```

## 清理构建目录

```bash
rm -rf build
```

然后重新执行配置和编译命令即可。
