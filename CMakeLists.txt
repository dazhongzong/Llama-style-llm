cmake_minimum_required(VERSION 3.18)

project(CudaLearning LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CTest)

option(BUILD_NVBENCH
    "Deprecated compatibility option; build benchmarks from benchmarks/"
    OFF)

if(BUILD_NVBENCH)
    message(FATAL_ERROR
        "NVBench 已改为独立工程，请使用: cmake -S benchmarks -B build-benchmarks")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 可在配置时覆盖：
# cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89
set(CMAKE_CUDA_ARCHITECTURES 80 89 CACHE STRING "CUDA architectures")

find_package(CUDAToolkit REQUIRED)

# CUDA 源文件：
#   推荐放在 src/kernels/
#   当前项目已有的 cuda/cuda_test.cu 也保留兼容
file(GLOB_RECURSE CUDA_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/cuda/*.cu"
)

# C++ 源文件放在 src/，例如 src/main.cpp、src/ops/*.cpp
file(GLOB_RECURSE CXX_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx"
)

set(APP_MAIN "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
set(LIBRARY_CXX_SOURCES ${CXX_SOURCES})
list(REMOVE_ITEM LIBRARY_CXX_SOURCES "${APP_MAIN}")

# 头文件只加入 IDE 工程显示，不会单独编译
file(GLOB_RECURSE PROJECT_HEADERS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/inc/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/inc/*.hh"
    "${CMAKE_CURRENT_SOURCE_DIR}/inc/*.hpp"
)

if(NOT CUDA_SOURCES)
    message(FATAL_ERROR "没有找到 CUDA 源文件，请将 .cu 文件放到 src/kernels/ 或 cuda/ 中。")
endif()

if(NOT EXISTS "${APP_MAIN}")
    message(FATAL_ERROR "没有找到 src/main.cpp。")
endif()

# CUDA 源文件编译为静态库
add_library(kernels STATIC
    ${CUDA_SOURCES}
    ${LIBRARY_CXX_SOURCES}
    ${PROJECT_HEADERS}
)

target_include_directories(kernels
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/inc"
)

set_target_properties(kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(kernels
    PUBLIC
        CUDA::cudart
)

# C++ 主程序
add_executable(app
    "${APP_MAIN}"
    ${PROJECT_HEADERS}
)

target_include_directories(app
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/inc"
)

target_link_libraries(app
    PRIVATE
        kernels
)

# 单元测试：tests/ 下每个源文件对应一个 CTest 测试。
if(BUILD_TESTING)
    file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cxx"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cu"
    )

    # bench_*.cpp 属于旧的手写 benchmark，不作为单元测试注册。
    list(FILTER TEST_SOURCES EXCLUDE REGEX "/tests/bench_[^/]*\\.(cpp|cc|cxx|cu)$")

    foreach(TEST_SOURCE IN LISTS TEST_SOURCES)
        get_filename_component(TEST_NAME "${TEST_SOURCE}" NAME_WE)
        set(TEST_TARGET "test_${TEST_NAME}")

        add_executable(${TEST_TARGET}
            "${TEST_SOURCE}"
            ${PROJECT_HEADERS}
        )

        target_include_directories(${TEST_TARGET}
            PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}/inc"
        )

        target_link_libraries(${TEST_TARGET}
            PRIVATE
                kernels
        )

        add_test(
            NAME ${TEST_TARGET}
            COMMAND ${TEST_TARGET}
        )
    endforeach()
endif()
