#define EIGEN_FAST_MATH 0
#include <ATen/ATen.h>
#include <Eigen/Core>
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

#define BM_BenchATenOp(name, op)                                               \
  static void BM_ATen##name(benchmark::State &state) {                         \
    at::set_num_threads(1);                                                    \
    for (auto _ : state) {                                                     \
      state.PauseTiming();                                                     \
      benchmark::ClobberMemory();                                              \
      at::Tensor a = at::rand(at::CPU(at::kFloat), {state.range(0)});          \
      at::Tensor b = at::rand(at::CPU(at::kFloat), {state.range(0)});          \
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
// BM_BenchReduceOp(sum);
// BM_BenchReduceOp(prod);
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
//   Eigen::ArrayXf a = Eigen::ArrayXf::Random(10);
//   std::cout << typeid(a.sum()).name() << std::endl;
// }
