#define EIGEN_FAST_MATH 0
#include<Eigen/Core>
#include <iostream>
#include <ATen/ATen.h>
#include <benchmark/benchmark.h>
#include <sleef.h>

constexpr size_t _ALIGNMENT = 32;

#define SETTING                                                                \
  ->Unit(benchmark::kMillisecond)                                              \
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

#define BM_BenchReduceOp(op)                                                   \
  static void BM_ATen##op(benchmark::State &state) {                           \
    at::set_num_threads(1);                                                    \
    for (auto _ : state) {                                                     \
      state.PauseTiming();                                                     \
      benchmark::ClobberMemory();                                              \
      at::Tensor a = at::rand(at::CPU(at::kFloat), {state.range(0)});          \
      benchmark::ClobberMemory();                                              \
      state.ResumeTiming();                                                    \
      for (int j = 0; j < state.range(1); ++j) {                               \
        at::Tensor b = a.op();                                                 \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  static void BM_Eigen##op(benchmark::State &state) {                          \
    for (auto _ : state) {                                                     \
      state.PauseTiming();                                                     \
      benchmark::ClobberMemory();                                              \
      Eigen::ArrayXf a = Eigen::ArrayXf::Random(state.range(0));               \
      float b;                                                                 \
      benchmark::ClobberMemory();                                              \
      state.ResumeTiming();                                                    \
      for (int j = 0; j < state.range(1); ++j) {                               \
        b = a.op();                                                            \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  BENCHMARK(BM_Eigen##op)SETTING;        \
  BENCHMARK(BM_ATen##op)SETTING;

#define BM_BenchOp(op)                                                         \
  static void BM_ATen##op(benchmark::State &state) {                           \
    at::set_num_threads(1);                                                    \
    for (auto _ : state) {                                                     \
      state.PauseTiming();                                                     \
      benchmark::ClobberMemory();                                              \
      at::Tensor a = at::rand(at::CPU(at::kFloat), {state.range(0)});          \
      at::Tensor b = at::rand(at::CPU(at::kFloat), {state.range(0)});          \
      benchmark::ClobberMemory();                                              \
      state.ResumeTiming();                                                    \
      for (int j = 0; j < state.range(1); ++j)                                 \
        b = a.op();                                                            \
    }                                                                          \
  }                                                                            \
  static void BM_Eigen##op(benchmark::State &state) {                          \
    for (auto _ : state) {                                                     \
      state.PauseTiming();                                                     \
      benchmark::ClobberMemory();                                              \
      Eigen::ArrayXf a = Eigen::ArrayXf::Random(state.range(0));               \
      Eigen::ArrayXf b = Eigen::ArrayXf::Random(state.range(0));               \
      benchmark::ClobberMemory();                                              \
      state.ResumeTiming();                                                    \
      for (int j = 0; j < state.range(1); ++j) {                               \
        b = a.op();                                                            \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  BENCHMARK(BM_ATen##op)SETTING;         \
  BENCHMARK(BM_Eigen##op)SETTING;

#define BM_BenchWithSleefOp(op)                                                \
  static void BM_Sleef##op(benchmark::State &state) {                          \
    for (auto _ : state) {                                                     \
      state.PauseTiming();                                                     \
      int64_t size = state.range(0);                                           \
      float *a_ptr;                                                            \
      int ret =                                                                \
          posix_memalign((void **)&a_ptr, _ALIGNMENT, size * sizeof(float));   \
      if (!ret)                                                                \
        assert(false);                                                         \
      int64_t vec_size = 8;                                                    \
      assert(state.range(0) % _ALIGNMENT == 0);                                \
      state.ResumeTiming();                                                    \
      for (int j = 0; j < state.range(1); ++j) {                               \
        int64_t d = 0;                                                         \
        for (; d < size - (size % vec_size); d += vec_size) {                  \
          __m256 values = _mm256_load_ps(a_ptr + d);                           \
          values = Sleef_##op##f8_u10(values);                                 \
          _mm256_store_ps(a_ptr + d, values);                                  \
        }                                                                      \
        benchmark::ClobberMemory();                                            \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  BM_BenchOp(op);                                                              \
  BENCHMARK(BM_Sleef##op)SETTING;


// Commented out means not supported
// TODO: Add comparison between intrinsics and ATen
BM_BenchWithSleefOp(acos)
BM_BenchWithSleefOp(asin)
BM_BenchWithSleefOp(atan)
BM_BenchReduceOp(sum)
BM_BenchReduceOp(prod)
// BM_BenchOp(erf)
BM_BenchWithSleefOp(exp)
BM_BenchWithSleefOp(expm1)
BM_BenchWithSleefOp(log)
BM_BenchWithSleefOp(log10)
BM_BenchWithSleefOp(log1p)
// BM_BenchOp(log2)
BM_BenchOp(ceil)
BM_BenchOp(floor)
BM_BenchOp(round)
BM_BenchOp(sqrt)
BM_BenchWithSleefOp(tanh)
// BM_BenchOp(trunc)

BENCHMARK_MAIN();
