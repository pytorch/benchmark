cmake_minimum_required (VERSION 3.0)
project (myproject C CXX)
find_package (Threads)
set(CMAKE_MODULE_PATH "${CMAKE_HOME_DIRECTORY}/cmake/Modules")
set(CMAKE_BINARY_DIR "build/bin")
set (CMAKE_CXX_STANDARD 11)
set(Eigen3_DIR "${CMAKE_HOME_DIRECTORY}/build/eigen")
find_package (Eigen3 REQUIRED NO_MODULE)
set(GBENCHMARK_INCLUDE "${CMAKE_HOME_DIRECTORY}/build/gbenchmark_install/include")
set(GBENCHMARK_LIB
    "${CMAKE_HOME_DIRECTORY}/build/gbenchmark_install/lib/libbenchmark_main.a"
    "${CMAKE_HOME_DIRECTORY}/build/gbenchmark_install/lib/libbenchmark.a")
set (CAFFE2_LIBRARY "${PYTORCH_HOME}/torch/lib/tmp_install/lib/libcaffe2.so")
set (CAFFE2_INCLUDE "${PYTORCH_HOME}/torch/lib/tmp_install/include")

set(MKL_LIBS "$ENV{CONDA_PREFIX}/lib/libmkl_gf_lp64.so" "$ENV{CONDA_PREFIX}/lib/libmkl_core.so" "$ENV{CONDA_PREFIX}/lib/libmkl_gnu_thread.so")
set(CONDA_INCLUDE "$ENV{CONDA_PREFIX}/include")
set(CONDA_LIBS "$ENV{CONDA_PREFIX}/lib/libiomp5.so")
message("CONDA_INCLUDE: ${CONDA_INCLUDE}")
message("CONDA_LIBS: ${CONDA_LIBS}")

include("${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/tbb/cmake/TBBGet.cmake")
tbb_get(TBB_ROOT tbb_root RELEASE_TAG 2018_U4 CONFIG_DIR TBB_DIR)
find_package(TBB REQUIRED)

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

include_directories (SYSTEM "${CAFFE2_INCLUDE}" "${GBENCHMARK_INCLUDE}" "${CONDA_INCLUDE}")


SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O3 -mavx2 -fopenmp")

add_executable (compare_eigen benchmarks/compare_eigen.cpp)

target_link_libraries(compare_eigen ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries (compare_eigen Eigen3::Eigen "${CAFFE2_LIBRARY}" "${GBENCHMARK_LIB}")
target_link_libraries(compare_eigen ${MKL_LIBS})
target_link_libraries(compare_eigen ${CONDA_LIBS})

add_executable (tbb_vs_omp benchmarks/tbb_vs_omp.cpp)

target_link_libraries(tbb_vs_omp ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(tbb_vs_omp TBB::tbb "${GBENCHMARK_LIB}")
target_link_libraries(tbb_vs_omp ${CONDA_LIBS})

add_executable (avx_sum benchmarks/avx_sum.cpp)

target_link_libraries(avx_sum ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(avx_sum TBB::tbb "${GBENCHMARK_LIB}")
target_link_libraries(avx_sum ${CONDA_LIBS})
