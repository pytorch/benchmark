cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(jit-bench)

find_package(Torch REQUIRED)

add_executable(jit-bench bench.cpp functions.cpp)
target_link_libraries(jit-bench "${TORCH_LIBRARIES}")
set_property(TARGET jit-bench PROPERTY CXX_STANDARD 11)
