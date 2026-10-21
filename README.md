# CudaLearning

一个使用 CUDA/C++ 实现的 MiniLLM 推理项目，包含 Transformer 推理、CUDA
算子、`tokenizers-cpp` 分词、单元测试和 NVBench benchmark。

## 项目结构

```text
inc/                 C++ 和 CUDA 头文件
src/                 C++ 源文件
src/kernels/         CUDA 源文件
tests/               单元测试
benchmarks/          NVBench benchmark
configs/             模型配置
third_party/         Git 子模块依赖
```

## 环境要求

- CMake >= 3.19（benchmark 工程需要 >= 3.30.4）
- 支持 C++20 的编译器
- CUDA Toolkit 和 NVIDIA 驱动
- Rust/Cargo（用于构建 `tokenizers-cpp`）
- `uv`（仅 Python 参考脚本需要）

初始化依赖：

```bash
git submodule update --init --recursive
```

## 构建主程序

在项目根目录执行：

```bash
cmake -S . -B build \
    -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --target app --parallel
```

模型文件放在 `models/` 下，并在 `configs/models.json` 中配置路径。每个模型
目录至少需要：`config.json`、`model.safetensors` 和 `tokenizer.json`。

## 文本生成

直接传入 prompt：

```bash
./build/app \
    --model smollm2-135m \
    --prompt "Explain CUDA kernels in one sentence." \
    --max-new-tokens 64
```

从标准输入读取：

```bash
echo "Explain CUDA kernels in one sentence." | \
    ./build/app --prompt - --max-new-tokens 64
```

持续对话：

```bash
./build/app \
    --model tinyllama-1.1b-chat \
    --chat \
    --max-new-tokens 64 \
    --temperature 0.7 \
    --top-k 50 \
    --top-p 0.95
```

`--temperature 0` 可以切换回确定性的 Greedy Decoding；`--seed` 用于复现采样结果。

聊天命令：

- `/help`：显示帮助
- `/reset`：清空历史
- `/exit` 或 `/quit`：退出

## 单元测试

```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

只运行一个测试：

```bash
ctest --test-dir build -R test_core --output-on-failure
```

## NVBench Benchmark

benchmark 是独立的 CMake 工程：

```bash
cmake -S benchmarks -B build-benchmarks \
    -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-benchmarks \
    --target all_benchmarks --parallel
```

运行 Transformer decode benchmark：

```bash
./build-benchmarks/bench_transformer_decode
```

通过环境变量切换模型：

```bash
MINILLM_MODEL_DIR=models/TinyLlama-1.1B-Chat-v1.0 \
    ./build-benchmarks/bench_transformer_decode
```

已有结果：

- [SmolLM2-135M RTX 5070](benchmarks/results/smollm2-135m-rtx5070.md)
- [TinyLlama-1.1B RTX 5070](benchmarks/results/tinyllama-1.1b-rtx5070.md)

## Python 参考脚本

```bash
uv sync
uv run python tools/inspect_model.py
```
