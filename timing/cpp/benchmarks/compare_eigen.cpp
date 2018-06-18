#define EIGEN_FAST_MATH 0
#define EIGEN_USE_MKL_ALL 1
#include <ATen/ATen.h>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <benchmark/benchmark.h>
#include <iostream>
#include <sleef.h>
#include <typeinfo>
#include <stdexcept>

// TODO: Matrix and Tensor reductions / unary ops for non-cont. memory

constexpr size_t _ALIGNMENT = 32;

#define SETTING                                                                \
  ->Unit(benchmark::kMicrosecond)                                              \
      ->Args({32768, 64})                                                      \
      ->Args({65536, 64})                                                      \
      ->Args({131072, 64})                                                     \
      ->Args({262144, 64})                                                     \
      ->Args({524288, 64})                                                     \
      ->Args({1048576, 64})                                                    \
      ->Args({2097152, 64})                                                    \
      ->Args({4194304, 64})                                                    \
      ->Args({8388608, 64})                                                    \
      ->Args({64777264, 64});

// General benchmark setup: Allocate fresh memory every 64 iterations
// Call the op once to warmup.

#define BM_BenchATenReduceOp(name, op, dim)                                    \
  static void BM_ATenReduce##dim##name(benchmark::State &state) {              \
    at::set_num_threads(1);                                                    \
    for (auto _ : state) {                                                     \
      state.PauseTiming();                                                     \
      double size_ = std::cbrt((double)(state.range(0)));                      \
      int size = (int)(size_);                                                 \
      benchmark::ClobberMemory();                                              \
      at::Tensor a = at::rand({size, size, size}, at::CPU(at::kFloat));        \
      at::Tensor b = at::rand({size, size}, at::CPU(at::kFloat));              \
      op;                                                                      \
      benchmark::ClobberMemory();                                              \
      state.ResumeTiming();                                                    \
      for (int j = 0; j < state.range(1); ++j) {                               \
        op;                                                                    \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  BENCHMARK(BM_ATenReduce##dim##name) SETTING;

#define BM_BenchATenOp(name, op)                                               \
  static void BM_ATen##name(benchmark::State &state) {                         \
    at::set_num_threads(1);                                                    \
    for (auto _ : state) {                                                     \
      state.PauseTiming();                                                     \
      benchmark::ClobberMemory();                                              \
      at::Tensor a = at::rand({state.range(0)}, at::CPU(at::kFloat));          \
      at::Tensor b = at::rand({state.range(0)}, at::CPU(at::kFloat));          \
      at::Tensor c;                                                            \
      op;                                                                      \
      benchmark::ClobberMemory();                                              \
      state.ResumeTiming();                                                    \
      for (int j = 0; j < state.range(1); ++j) {                               \
        op;                                                                    \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  BENCHMARK(BM_ATen##name) SETTING;

#define BM_BenchEigenReduceOp(name, op, dim)                                   \
  static void BM_EigenReduce##dim##name(benchmark::State &state) {             \
    for (auto _ : state) {                                                     \
      state.PauseTiming();                                                     \
      benchmark::ClobberMemory();                                              \
      double size_ = std::cbrt((double)(state.range(0)));                      \
      int size = (int)(size_);                                                 \
      Eigen::array<int, 1> dims({dim /* dimension to reduce */});              \
      Eigen::Tensor<float, 3> a(size, size, size);                             \
      Eigen::Tensor<float, 2> b(size, size);                                   \
      a.setRandom();                                                           \
      b.setRandom();                                                           \
      op;                                                                      \
      benchmark::ClobberMemory();                                              \
      state.ResumeTiming();                                                    \
      for (int j = 0; j < state.range(1); ++j) {                               \
        op;                                                                    \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  BENCHMARK(BM_EigenReduce##dim##name) SETTING;

#define BM_BenchEigenOp(name, op)                                              \
  static void BM_Eigen##name(benchmark::State &state) {                        \
    for (auto _ : state) {                                                     \
      state.PauseTiming();                                                     \
      benchmark::ClobberMemory();                                              \
      Eigen::ArrayXf a = Eigen::ArrayXf::Random(state.range(0));               \
      Eigen::ArrayXf b = Eigen::ArrayXf::Random(state.range(0));               \
      float c;                                                                 \
      op;                                                                      \
      benchmark::ClobberMemory();                                              \
      state.ResumeTiming();                                                    \
      for (int j = 0; j < state.range(1); ++j) {                               \
        op;                                                                    \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  BENCHMARK(BM_Eigen##name) SETTING;

#define BM_BenchReduceOp(op)                                                   \
  BM_BenchEigenReduceOp(_reduce_##op, b = a.op(dims), 0);                      \
  BM_BenchATenReduceOp(_reduce_##op, b = a.op(0), 0);                          \
  BM_BenchEigenReduceOp(_reduce_##op, b = a.op(dims), 1);                      \
  BM_BenchATenReduceOp(_reduce_##op, b = a.op(1), 1);                          \
  BM_BenchEigenReduceOp(_reduce_##op, b = a.op(dims), 2);                      \
  BM_BenchATenReduceOp(_reduce_##op, b = a.op(2), 2);                          \
  BM_BenchEigenOp(_reduce_##op, c = a.op());                                   \
  BM_BenchATenOp(_reduce_##op, c = a.op());

#define BM_BenchUnaryOp(op)                                                    \
  BM_BenchATenOp(_unary_##op, at::op##_out(b, a));                             \
  BM_BenchEigenOp(_unary_##op, b = a.op());

#define BM_BenchUnaryWithSleefOp(op)                                           \
  static void BM_Sleef_##op(benchmark::State &state) {                         \
    for (auto _ : state) {                                                     \
      state.PauseTiming();                                                     \
      benchmark::ClobberMemory();                                              \
      int64_t size = state.range(0);                                           \
      float *a_ptr;                                                            \
      float *b_ptr;                                                            \
      int ret =                                                                \
          posix_memalign((void **)&a_ptr, _ALIGNMENT, size * sizeof(float));   \
      ret |=                                                                   \
          posix_memalign((void **)&b_ptr, _ALIGNMENT, size * sizeof(float));   \
      if (ret)                                                                 \
        throw std::runtime_error("memory align failed");                       \
      int64_t vec_size = 8;                                                    \
      assert(state.range(0) % _ALIGNMENT == 0);                                \
      benchmark::ClobberMemory();                                              \
      state.ResumeTiming();                                                    \
      for (int j = 0; j < state.range(1); ++j) {                               \
        int64_t d = 0;                                                         \
        for (; d < size - (size % vec_size); d += vec_size) {                  \
          __m256 values = _mm256_load_ps(a_ptr + d);                           \
          values = Sleef_##op##f8_u10(values);                                 \
          _mm256_store_ps(b_ptr + d, values);                                  \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  BENCHMARK(BM_Sleef_##op) SETTING;                                            \
  BM_BenchUnaryOp(op);

// Commented out means not supported
// TODO: Add comparison between intrinsics and ATen
BM_BenchReduceOp(sum);
BM_BenchReduceOp(prod);
BM_BenchReduceOp(mean);
BM_BenchUnaryWithSleefOp(exp);
BM_BenchUnaryWithSleefOp(log);
BM_BenchUnaryWithSleefOp(acos);
BM_BenchUnaryWithSleefOp(asin);
BM_BenchUnaryWithSleefOp(atan);
// BM_BenchOp(erf);
BM_BenchUnaryWithSleefOp(expm1);
BM_BenchUnaryWithSleefOp(log10);
BM_BenchUnaryWithSleefOp(log1p);
// BM_BenchOp(log2);
BM_BenchUnaryOp(ceil);
BM_BenchUnaryOp(floor);
BM_BenchUnaryOp(round);
BM_BenchUnaryOp(sqrt);
BM_BenchUnaryWithSleefOp(tanh);
// BM_BenchOp(trunc);

BENCHMARK_MAIN();

// int main() {
// // Create a tensor of 2 dimensions
// Eigen::Tensor<int, 2> a(2, 3);
// a.setValues({{1, 2, 3}, {6, 5, 4}});
// // Reduce it along the second dimension (1)...
// Eigen::array<int, 1> dims({1 /* dimension to reduce */});
// // ...using the "maximum" operator.
// // The result is a tensor with one dimension.  The size of
// // that dimension is the same as the first (non-reduced) dimension of a.
// Eigen::Tensor<int, 1> b = a.sum(dims);
// std::cout << "a" << std::endl << a << std::endl << std::endl;
// std::cout << "b" << std::endl << b << std::endl << std::endl;
//   // Eigen::ArrayXf a = Eigen::ArrayXf::Random(10);
//   // std::cout << typeid(a.sum()).name() << std::endl;
// }
