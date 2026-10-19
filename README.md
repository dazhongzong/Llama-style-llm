# CudaLearning

这是一个 CUDA/C++ 学习项目。项目使用 CMake 构建，CUDA 算子编译为静态库，C++ 主程序和单元测试链接该静态库。

## 目录结构

```text
inc/                 C++ 头文件
inc/kernels/         CUDA 头文件
src/                 C++ 源文件，例如 main.cpp、ops/*.cpp
src/kernels/         推荐放置 CUDA 源文件
cuda/                当前项目已有的 CUDA 源文件，CMake 也会自动收集
tests/               单元测试，每个源文件生成一个测试程序
benchmarks/          NVBench benchmark 源文件
```

## 环境要求

- CMake >= 3.18
- C++ 编译器，支持 C++20
- CUDA Toolkit，当前配置使用 CUDA 17 语言标准
- 可用的 NVIDIA 驱动和 CUDA 设备（运行 CUDA 测试时需要）

检查工具：

```bash
cmake --version
nvcc --version
g++ --version
```

## 配置项目

在项目根目录执行：

```bash
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
