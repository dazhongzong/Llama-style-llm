cmake_minimum_required(VERSION 3.30.4)

project(CudaLearningBenchmarks LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 80 89 CACHE STRING
    "CUDA architectures used by benchmarks")

set(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")

find_package(CUDAToolkit REQUIRED)
include(FetchContent)

# NVBench 及其依赖只属于 benchmark 工程。
FetchContent_Declare(nvbench
    GIT_REPOSITORY https://github.com/NVIDIA/nvbench.git
    GIT_TAG main
    GIT_SHALLOW TRUE
)

# 已下载的依赖不在每次构建时访问 GitHub。
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL
    "Do not update already fetched dependencies" FORCE)

set(NVBench_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(NVBench_ENABLE_TESTING OFF CACHE BOOL "" FORCE)

# 如果之前通过 -DFETCHCONTENT_SOURCE_DIR_* 使用过本地依赖目录，
# 但该目录后来被删除，则清除失效缓存，让 FetchContent 重新下载。
foreach(DEPENDENCY IN ITEMS NVBENCH RAPIDS-CMAKE FMT NLOHMANN_JSON)
    set(SOURCE_DIR_VARIABLE "FETCHCONTENT_SOURCE_DIR_${DEPENDENCY}")

    if(DEFINED ${SOURCE_DIR_VARIABLE})
        set(SOURCE_DIR_VALUE "${${SOURCE_DIR_VARIABLE}}")

        if(NOT EXISTS "${SOURCE_DIR_VALUE}")
            message(STATUS
                "Ignoring missing ${SOURCE_DIR_VARIABLE}: ${SOURCE_DIR_VALUE}")
            unset(${SOURCE_DIR_VARIABLE} CACHE)
        endif()
    endif()
endforeach()

FetchContent_MakeAvailable(nvbench)

# benchmark 需要链接项目中的算子实现，但不会依赖根工程的 CMake target。
file(GLOB_RECURSE CUDA_SOURCES CONFIGURE_DEPENDS
    "${PROJECT_ROOT}/src/kernels/*.cu"
    "${PROJECT_ROOT}/cuda/*.cu"
)

file(GLOB_RECURSE CXX_SOURCES CONFIGURE_DEPENDS
    "${PROJECT_ROOT}/src/*.cpp"
    "${PROJECT_ROOT}/src/*.cc"
    "${PROJECT_ROOT}/src/*.cxx"
)

list(FILTER CXX_SOURCES EXCLUDE REGEX "/src/main\\.(cpp|cc|cxx)$")

file(GLOB_RECURSE PROJECT_HEADERS CONFIGURE_DEPENDS
    "${PROJECT_ROOT}/inc/*.h"
    "${PROJECT_ROOT}/inc/*.hh"
    "${PROJECT_ROOT}/inc/*.hpp"
)

add_library(benchmark_kernels STATIC
    ${CUDA_SOURCES}
    ${CXX_SOURCES}
    ${PROJECT_HEADERS}
)

target_include_directories(benchmark_kernels
    PUBLIC
        "${PROJECT_ROOT}/inc"
)

set_target_properties(benchmark_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(benchmark_kernels
    PUBLIC
        CUDA::cudart
)

file(GLOB_RECURSE BENCHMARK_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
)

foreach(BENCHMARK_SOURCE IN LISTS BENCHMARK_SOURCES)
    get_filename_component(BENCHMARK_NAME "${BENCHMARK_SOURCE}" NAME_WE)

    add_executable(${BENCHMARK_NAME}
        "${BENCHMARK_SOURCE}"
    )

    target_include_directories(${BENCHMARK_NAME}
        PRIVATE
            "${PROJECT_ROOT}/inc"
    )

    target_link_libraries(${BENCHMARK_NAME}
        PRIVATE
            benchmark_kernels
            nvbench::main
    )
endforeach()

add_custom_target(all_benchmarks)

foreach(BENCHMARK_SOURCE IN LISTS BENCHMARK_SOURCES)
    get_filename_component(BENCHMARK_NAME "${BENCHMARK_SOURCE}" NAME_WE)
    add_dependencies(all_benchmarks ${BENCHMARK_NAME})
endforeach()
