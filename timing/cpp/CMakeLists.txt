cmake_minimum_required (VERSION 3.0)
project (myproject C CXX)
find_package (Threads)
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O3 -mavx2 -funroll-loops")
set(CMAKE_BINARY_DIR "build/bin")
set (CMAKE_CXX_STANDARD 11)
set(Eigen3_DIR "${CMAKE_HOME_DIRECTORY}/build/eigen")
find_package (Eigen3 REQUIRED NO_MODULE)
set(GBENCHMARK_INCLUDE "${CMAKE_HOME_DIRECTORY}/build/benchmark_install/include")
set(GBENCHMARK_LIB
    "${CMAKE_HOME_DIRECTORY}/build/benchmark_install/lib/libbenchmark_main.a"
    "${CMAKE_HOME_DIRECTORY}/build/benchmark_install/lib/libbenchmark.a")
set (CAFFE2_LIBRARY "${PYTORCH_HOME}/torch/lib/tmp_install/lib/libcaffe2.so")
set (CAFFE2_INCLUDE "${PYTORCH_HOME}/torch/lib/tmp_install/include")

# Preserve values for the main build
set(__aten_sleef_build_shared_libs ${BUILD_SHARED_LIBS})
set(__aten_sleef_build_tests ${BUILD_TESTS})

# Unset our restrictive C++ flags here and reset them later.
# Remove this once we use proper target_compile_options.
set(OLD_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
set(CMAKE_CXX_FLAGS)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build sleef static" FORCE)
set(BUILD_DFT OFF CACHE BOOL "Don't build sleef DFT lib" FORCE)
set(BUILD_GNUABI_LIBS OFF CACHE BOOL "Don't build sleef gnuabi libs" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "Don't build sleef tests" FORCE)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/sleef" ${CMAKE_BINARY_DIR}/sleef)
link_directories(${CMAKE_BINARY_DIR}/sleef/lib)

set(CMAKE_CXX_FLAGS ${OLD_CMAKE_CXX_FLAGS})

# Set these back. TODO: Use SLEEF_ to pass these instead
set(BUILD_SHARED_LIBS ${__aten_sleef_build_shared_libs} CACHE BOOL "Build shared libs" FORCE)
set(BUILD_TESTS ${__aten_sleef_build_tests} CACHE BOOL "Build tests" FORCE)

include_directories (SYSTEM "${CAFFE2_INCLUDE}" "${GBENCHMARK_INCLUDE}")
add_executable (example benchmarks/compare_eigen.cpp)
target_link_libraries (example Eigen3::Eigen "${CAFFE2_LIBRARY}" "${GBENCHMARK_LIB}")
target_link_libraries(example ${CMAKE_THREAD_LIBS_INIT})
