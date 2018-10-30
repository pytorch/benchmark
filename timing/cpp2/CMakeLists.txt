cmake_minimum_required (VERSION 3.0)
project (myproject C CXX)
find_package (Threads)
set(CMAKE_MODULE_PATH "${CMAKE_HOME_DIRECTORY}/cmake/Modules")
set(CMAKE_BINARY_DIR "build/bin")
set (CMAKE_CXX_STANDARD 11)
set(GBENCHMARK_INCLUDE "${CMAKE_HOME_DIRECTORY}/build/gbenchmark_install/include")
set(GBENCHMARK_LIB
    "${CMAKE_HOME_DIRECTORY}/build/gbenchmark_install/lib/libbenchmark_main.a"
    "${CMAKE_HOME_DIRECTORY}/build/gbenchmark_install/lib/libbenchmark.a")

set(OLD_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
set(CMAKE_CXX_FLAGS)
set(CMAKE_CXX_FLAGS ${OLD_CMAKE_CXX_FLAGS})

include_directories (SYSTEM "${GBENCHMARK_INCLUDE}")

SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O3 -mavx2 -fopenmp")

find_package(Torch REQUIRED)

add_executable (aten_overheads benchmarks/aten_overheads.cpp)
target_link_libraries(aten_overheads ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(aten_overheads "${GBENCHMARK_LIB}")
target_link_libraries(aten_overheads "${TORCH_LIBRARIES}")

add_executable (tensor_properties benchmarks/tensor_properties.cpp)
target_link_libraries(tensor_properties ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(tensor_properties "${GBENCHMARK_LIB}")
target_link_libraries(tensor_properties "${TORCH_LIBRARIES}")

add_executable (torch_empty benchmarks/torch_empty.cpp)
target_link_libraries(torch_empty ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(torch_empty "${GBENCHMARK_LIB}")
target_link_libraries(torch_empty "${TORCH_LIBRARIES}")

add_executable (tensor_shape benchmarks/tensor_shape.cpp)
target_link_libraries(tensor_shape ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(tensor_shape "${GBENCHMARK_LIB}")
target_link_libraries(tensor_shape "${TORCH_LIBRARIES}")

add_executable (tensor_allocation benchmarks/tensor_allocation.cpp)
target_link_libraries(tensor_allocation ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(tensor_allocation "${GBENCHMARK_LIB}")
target_link_libraries(tensor_allocation "${TORCH_LIBRARIES}")
