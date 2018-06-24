#define EIGEN_FAST_MATH 0
#define EIGEN_USE_MKL_ALL 1
#include <ATen/ATen.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <benchmark/benchmark.h>
#include <iostream>
#include <omp.h>
#include <sleef.h>
#include <stdexcept>
#include <typeinfo>

// Mimic TH alignment
constexpr size_t _ALIGNMENT = 64;

void make_vector(float *data_, size_t size) {
  for (size_t i = 0; i < size; i++) {
    data_[i] = (float)(i % 1024);
  }
}

void make_float_data(float **data_, size_t size) {
  if (posix_memalign((void **)data_, _ALIGNMENT, size * sizeof(float)))
    throw std::invalid_argument("received negative value");
  memset(*data_, 0, size * sizeof(float));
}

float get_random_value() {
  std::random_device
      rd; // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<> dis(1, 6);
  return dis(gen);
}

float do_something(float a) {
  benchmark::DoNotOptimize(a = a * a);
  return a;
}

using InnerStride = Eigen::InnerStride<Eigen::Dynamic>;

template <typename T>
using EigenMatrixMap =
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>,
               Eigen::AlignmentType::Aligned64,
               Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;

template <typename T>
using EigenVectorMap =
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>,
               Eigen::AlignmentType::Aligned64,
               Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;

// TODO: Use Stride class

#define SETTING                                                                \
  ->Args({32768, 64})                                                          \
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
//
// TODO: Eigen doesn't parallelize reductions?

#define BM_BenchATenReduceOp(name, op, dim, stride)                            \
  static void BM_ATenReduce##dim##stride##name(benchmark::State &state) {      \
    for (auto _ : state) {                                                     \
      state.PauseTiming();                                                     \
      benchmark::ClobberMemory();                                              \
      double size_ = std::sqrt((double)(state.range(0)));                      \
      int size = (int)(size_);                                                 \
      state.counters["stride"] = stride;                                       \
      state.counters["dim"] = dim;                                             \
      state.counters["size"] = size;                                           \
      state.counters["iter"] = state.range(1);                                 \
      benchmark::ClobberMemory();                                              \
      at::Tensor a =                                                           \
          at::rand({size, size, stride}, at::CPU(at::kFloat)).select(2, 0);    \
      at::Tensor b =                                                           \
          at::rand({size, stride}, at::CPU(at::kFloat)).select(1, 0);          \
      op;                                                                      \
      benchmark::ClobberMemory();                                              \
      benchmark::ClobberMemory();                                              \
      state.ResumeTiming();                                                    \
      for (int j = 0; j < state.range(1); ++j) {                               \
        op;                                                                    \
      }                                                                        \
      state.PauseTiming();                                                     \
      op;                                                                      \
      benchmark::ClobberMemory();                                              \
      benchmark::ClobberMemory();                                              \
      state.ResumeTiming();                                                    \
    }                                                                          \
  }                                                                            \
  BENCHMARK(BM_ATenReduce##dim##stride##name) SETTING;

#define BM_BenchATenOp(name, op, stride)                                       \
  static void BM_ATen##stride##name(benchmark::State &state) {                 \
    for (auto _ : state) {                                                     \
      state.PauseTiming();                                                     \
      benchmark::ClobberMemory();                                              \
      state.counters["stride"] = stride;                                       \
      state.counters["dim"] = -1;                                              \
      state.counters["size"] = state.range(0);                                 \
      state.counters["iter"] = state.range(1);                                 \
      at::Tensor a = at::rand({state.range(0), stride}, at::CPU(at::kFloat))   \
                         .select(1, 0);                                        \
      at::Tensor b = at::rand({state.range(0), stride}, at::CPU(at::kFloat))   \
                         .select(1, 0);                                        \
      at::Tensor c;                                                            \
      op;                                                                      \
      benchmark::ClobberMemory();                                              \
      benchmark::ClobberMemory();                                              \
      state.ResumeTiming();                                                    \
      for (int j = 0; j < state.range(1); ++j) {                               \
        op;                                                                    \
      }                                                                        \
      state.PauseTiming();                                                     \
      op;                                                                      \
      benchmark::ClobberMemory();                                              \
      benchmark::ClobberMemory();                                              \
      state.ResumeTiming();                                                    \
    }                                                                          \
  }                                                                            \
  BENCHMARK(BM_ATen##stride##name) SETTING;

#define BM_BenchEigenReduceOp(name, op, dim, stride)                           \
  static void BM_EigenReduce##dim##name##stride(benchmark::State &state) {     \
    for (auto _ : state) {                                                     \
      state.PauseTiming();                                                     \
      double size_ = std::sqrt((double)(state.range(0)));                      \
      int size = (int)(size_);                                                 \
      state.counters["stride"] = stride;                                       \
      state.counters["dim"] = dim;                                             \
      state.counters["size"] = size;                                           \
      state.counters["iter"] = state.range(1);                                 \
      float *data_ = NULL;                                                     \
      make_float_data(&data_, size *size *stride);                             \
      make_vector(data_, size *size);                                          \
      float *out_data_ = NULL;                                                 \
      make_float_data(&out_data_, size *stride);                               \
      make_vector(out_data_, size);                                            \
      EigenMatrixMap<float> a(                                                 \
          data_, size, size,                                                   \
          Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(1, stride));           \
      EigenVectorMap<float> b(                                                 \
          out_data_, size,                                                     \
          Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(1, stride));           \
      op;                                                                      \
      benchmark::ClobberMemory();                                              \
      benchmark::ClobberMemory();                                              \
      state.ResumeTiming();                                                    \
      for (int j = 0; j < state.range(1); ++j) {                               \
        op;                                                                    \
      }                                                                        \
      state.PauseTiming();                                                     \
      op;                                                                      \
      free(data_);                                                             \
      free(out_data_);                                                         \
      benchmark::ClobberMemory();                                              \
      benchmark::ClobberMemory();                                              \
      state.ResumeTiming();                                                    \
    }                                                                          \
  }                                                                            \
  BENCHMARK(BM_EigenReduce##dim##name##stride) SETTING;

#define BM_BenchEigenOp(name, op, stride)                                      \
  static void BM_Eigen##name##stride(benchmark::State &state) {                \
    for (auto _ : state) {                                                     \
      state.PauseTiming();                                                     \
      int64_t size = state.range(0);                                           \
      state.counters["stride"] = stride;                                       \
      state.counters["dim"] = -1;                                              \
      state.counters["size"] = size;                                           \
      state.counters["iter"] = state.range(1);                                 \
      float *data_ = NULL;                                                     \
      make_float_data(&data_, size *stride);                                   \
      make_vector(data_, size);                                                \
      float *out_data_ = NULL;                                                 \
      make_float_data(&out_data_, size *stride);                               \
      make_vector(out_data_, size);                                            \
      EigenVectorMap<float> a(                                                 \
          data_, size,                                                         \
          Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(1, stride));           \
      EigenVectorMap<float> b(                                                 \
          out_data_, size,                                                     \
          Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(1, stride));           \
      float c = get_random_value();                                            \
      op;                                                                      \
      benchmark::ClobberMemory();                                              \
      benchmark::ClobberMemory();                                              \
      state.ResumeTiming();                                                    \
      for (int j = 0; j < state.range(1); ++j) {                               \
        op;                                                                    \
      }                                                                        \
      state.PauseTiming();                                                     \
      c = do_something(c);                                                     \
      op;                                                                      \
      free(data_);                                                             \
      free(out_data_);                                                         \
      benchmark::ClobberMemory();                                              \
      benchmark::ClobberMemory();                                              \
      state.ResumeTiming();                                                    \
    }                                                                          \
  }                                                                            \
  BENCHMARK(BM_Eigen##name##stride) SETTING;

#define BM_BenchReduceOp(op, stride)                                           \
  BM_BenchEigenOp(_reduce_##op, c += a.op(), stride);                          \
  BM_BenchEigenReduceOp(_reduce_##op, b = a.colwise().op(), 0, stride);        \
  BM_BenchEigenReduceOp(_reduce_##op, b = a.rowwise().op(), 1, stride);        \
  BM_BenchATenOp(_reduce_##op, c = a.op(), stride);                            \
  BM_BenchATenReduceOp(_reduce_##op, b = a.op(0), 0, stride);                  \
  BM_BenchATenReduceOp(_reduce_##op, b = a.op(1), 1, stride);

#define BM_BenchUnaryOp(op, stride)                                            \
  BM_BenchEigenOp(_unary_##op, b = a.array().op(), stride);                    \
  BM_BenchATenOp(_unary_##op, at::op##_out(b, a), stride);

#define BM_BenchUnaryWithSleefOp(op)                                           \
  static void BM_Sleef_##op(benchmark::State &state) {                         \
    for (auto _ : state) {                                                     \
      state.PauseTiming();                                                     \
      benchmark::ClobberMemory();                                              \
      int64_t size = state.range(0);                                           \
      state.counters["stride"] = 1;                                            \
      state.counters["dim"] = -1;                                              \
      state.counters["size"] = state.range(0);                                 \
      state.counters["iter"] = state.range(1);                                 \
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
  BM_BenchUnaryOp(op, 1);

// Commented out means not supported
// TODO: Add comparison between intrinsics and ATen
BM_BenchReduceOp(sum, 4);
BM_BenchReduceOp(prod, 4);
BM_BenchReduceOp(mean, 4);
BM_BenchReduceOp(sum, 1);
BM_BenchReduceOp(prod, 1);
BM_BenchReduceOp(mean, 1);
// BM_BenchReduceOp(sum, 16);
// BM_BenchReduceOp(prod, 16);
// BM_BenchReduceOp(mean, 16);
BM_BenchUnaryWithSleefOp(exp);
BM_BenchUnaryWithSleefOp(log);
// BM_BenchUnaryWithSleefOp(acos);
// BM_BenchUnaryWithSleefOp(asin);
// BM_BenchUnaryWithSleefOp(atan);
// BM_BenchOp(erf);
// BM_BenchUnaryWithSleefOp(expm1);
// BM_BenchUnaryWithSleefOp(log10);
// BM_BenchUnaryWithSleefOp(log1p);
// BM_BenchOp(log2);
BM_BenchUnaryOp(ceil, 1);
BM_BenchUnaryOp(floor, 1);
BM_BenchUnaryOp(ceil, 4);
BM_BenchUnaryOp(floor, 4);
// BM_BenchUnaryOp(round);
// BM_BenchUnaryOp(sqrt);
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
