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

// General benchmark setup: Allocate fresh memory every 64 iterations
// Call the op once to warmup, once after outside timings to maintain lifespan.

#define BM_BenchATenReduceOp(name, op)                                         \
  static void BM_ATen##name(benchmark::State &state, int64_t stride,           \
                            int64_t size__, int64_t iter) {                    \
    for (auto _ : state) {                                                     \
      state.PauseTiming();                                                     \
      benchmark::ClobberMemory();                                              \
      double size_ = std::sqrt((double)(size__));                              \
      int size = (int)(size_);                                                 \
      state.counters["stride"] = stride;                                       \
      state.counters["size"] = size;                                           \
      state.counters["iter"] = iter;                                           \
      benchmark::ClobberMemory();                                              \
      at::Tensor a =                                                           \
          at::rand({size, size, stride}, at::CPU(at::kFloat)).select(2, 0);    \
      at::Tensor b =                                                           \
          at::rand({size, stride}, at::CPU(at::kFloat)).select(1, 0);          \
      op;                                                                      \
      benchmark::ClobberMemory();                                              \
      benchmark::ClobberMemory();                                              \
      state.ResumeTiming();                                                    \
      for (int j = 0; j < iter; ++j) {                                         \
        op;                                                                    \
      }                                                                        \
      state.PauseTiming();                                                     \
      op;                                                                      \
      benchmark::ClobberMemory();                                              \
      benchmark::ClobberMemory();                                              \
      state.ResumeTiming();                                                    \
    }                                                                          \
  }

#define BM_BenchATenOp(name, op)                                               \
  static void BM_ATen##name(benchmark::State &state, int64_t stride,           \
                            int64_t size, int64_t iter) {                      \
    for (auto _ : state) {                                                     \
      state.PauseTiming();                                                     \
      benchmark::ClobberMemory();                                              \
      state.counters["stride"] = stride;                                       \
      state.counters["size"] = size;                                           \
      state.counters["iter"] = iter;                                           \
      at::Tensor a =                                                           \
          at::rand({size, stride}, at::CPU(at::kFloat)).select(1, 0);          \
      at::Tensor b =                                                           \
          at::rand({size, stride}, at::CPU(at::kFloat)).select(1, 0);          \
      at::Tensor c;                                                            \
      op;                                                                      \
      benchmark::ClobberMemory();                                              \
      benchmark::ClobberMemory();                                              \
      state.ResumeTiming();                                                    \
      for (int j = 0; j < iter; ++j) {                                         \
        op;                                                                    \
      }                                                                        \
      state.PauseTiming();                                                     \
      op;                                                                      \
      benchmark::ClobberMemory();                                              \
      benchmark::ClobberMemory();                                              \
      state.ResumeTiming();                                                    \
    }                                                                          \
  }

#define BM_BenchEigenReduceOp(name, op, dim)                                   \
  static void BM_Eigen##name(benchmark::State &state, int64_t stride,          \
                             int64_t size__, int64_t iter) {                   \
    for (auto _ : state) {                                                     \
      state.PauseTiming();                                                     \
      double size_ = std::sqrt((double)(size__));                              \
      int size = (int)(size_);                                                 \
      state.counters["stride"] = stride;                                       \
      state.counters["size"] = size;                                           \
      state.counters["iter"] = iter;                                           \
      float *data_ = NULL;                                                     \
      make_float_data(&data_, size *size *stride);                             \
      make_vector(data_, size *size *stride);                                  \
      float *out_data_ = NULL;                                                 \
      make_float_data(&out_data_, size *stride);                               \
      make_vector(out_data_, size *stride);                                    \
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
      for (int j = 0; j < iter; ++j) {                                         \
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
  }

#define BM_BenchEigenOp(name, op)                                              \
  static void BM_Eigen##name(benchmark::State &state, int64_t stride,          \
                             int64_t size, int64_t iter) {                     \
    for (auto _ : state) {                                                     \
      state.PauseTiming();                                                     \
      state.counters["stride"] = stride;                                       \
      state.counters["size"] = size;                                           \
      state.counters["iter"] = iter;                                           \
      float *data_ = NULL;                                                     \
      make_float_data(&data_, size *stride);                                   \
      make_vector(data_, size *stride);                                        \
      float *out_data_ = NULL;                                                 \
      make_float_data(&out_data_, size *stride);                               \
      make_vector(out_data_, size *stride);                                    \
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
      for (int j = 0; j < iter; ++j) {                                         \
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
  }

#define BM_BenchReduceOp(op)                                                   \
  BM_BenchEigenOp(_reduce_##op, c += a.op());                                  \
  BM_BenchEigenReduceOp(_reduce_colwise_##op, b = a.colwise().op(), 0);        \
  BM_BenchEigenReduceOp(_reduce_rowwise_##op, b = a.rowwise().op(), 1);        \
  BM_BenchATenOp(_reduce_##op, c = a.op());                                    \
  BM_BenchATenReduceOp(_reduce_colwise_##op, b = a.op(0));                     \
  BM_BenchATenReduceOp(_reduce_rowwise_##op, b = a.op(1));

#define BM_BenchUnaryOp(op)                                                    \
  BM_BenchEigenOp(_unary_##op, b = a.array().op());                            \
  BM_BenchATenOp(_unary_##op, at::op##_out(b, a));

#define BM_BenchUnaryWithSleefOp(op)                                           \
  static void BM_Sleef_##op(benchmark::State &state, int64_t size,             \
                            int64_t iter) {                                    \
    for (auto _ : state) {                                                     \
      state.PauseTiming();                                                     \
      benchmark::ClobberMemory();                                              \
      state.counters["stride"] = 1;                                            \
      state.counters["size"] = size;                                           \
      state.counters["iter"] = iter;                                           \
      float *a_ptr;                                                            \
      float *b_ptr;                                                            \
      int ret =                                                                \
          posix_memalign((void **)&a_ptr, _ALIGNMENT, size * sizeof(float));   \
      ret |=                                                                   \
          posix_memalign((void **)&b_ptr, _ALIGNMENT, size * sizeof(float));   \
      if (ret)                                                                 \
        throw std::runtime_error("memory align failed");                       \
      int64_t vec_size = 8;                                                    \
      assert(size % _ALIGNMENT == 0);                                          \
      benchmark::ClobberMemory();                                              \
      state.ResumeTiming();                                                    \
      for (int j = 0; j < iter; ++j) {                                         \
        int64_t d = 0;                                                         \
        for (; d < size - (size % vec_size); d += vec_size) {                  \
          __m256 values = _mm256_load_ps(a_ptr + d);                           \
          values = Sleef_##op##f8_u10(values);                                 \
          _mm256_store_ps(b_ptr + d, values);                                  \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  BM_BenchUnaryOp(op);

// Commented out means not supported
// TODO: Add comparison between intrinsics and ATen
BM_BenchReduceOp(sum);
BM_BenchReduceOp(prod);
BM_BenchUnaryWithSleefOp(exp);
BM_BenchUnaryWithSleefOp(log);
BM_BenchUnaryOp(floor);
// BM_BenchUnaryWithSleefOp(acos);
// BM_BenchUnaryWithSleefOp(asin);
// BM_BenchUnaryWithSleefOp(atan);
// BM_BenchOp(erf);
// BM_BenchUnaryWithSleefOp(expm1);
// BM_BenchUnaryWithSleefOp(log10);
// BM_BenchUnaryWithSleefOp(log1p);
// BM_BenchOp(log2);
// BM_BenchUnaryOp(round);
// BM_BenchUnaryOp(sqrt);
// BM_BenchOp(trunc);

// BENCHMARK_MAIN();

int main(int argc, char **argv) {
  int64_t iter = 64;
  for (int64_t size = 32768; size < 64777264; size *= 2) {
    benchmark::RegisterBenchmark("BM_Sleef_log", &BM_Sleef_log, size, iter);
    benchmark::RegisterBenchmark("BM_Sleef_exp", &BM_Sleef_exp, size, iter);
    for (int64_t stride = 1; stride < 16; stride *= 2) {
      benchmark::RegisterBenchmark("BM_Eigen_unary_log", &BM_Eigen_unary_log,
                                   stride, size, iter);
      benchmark::RegisterBenchmark("BM_Eigen_unary_exp", &BM_Eigen_unary_exp,
                                   stride, size, iter);
      benchmark::RegisterBenchmark("BM_Eigen_unary_floor",
                                   &BM_Eigen_unary_floor, stride, size, iter);

      benchmark::RegisterBenchmark("BM_ATen_unary_log", &BM_ATen_unary_log,
                                   stride, size, iter);
      benchmark::RegisterBenchmark("BM_ATen_unary_exp", &BM_ATen_unary_exp,
                                   stride, size, iter);
      benchmark::RegisterBenchmark("BM_ATen_unary_floor", &BM_ATen_unary_floor,
                                   stride, size, iter);

      benchmark::RegisterBenchmark("BM_Eigen_reduce_sum", &BM_Eigen_reduce_sum,
                                   stride, size, iter);
      benchmark::RegisterBenchmark("BM_Eigen_reduce_colwise_sum",
                                   &BM_Eigen_reduce_colwise_sum, stride, size,
                                   iter);
      benchmark::RegisterBenchmark("BM_Eigen_reduce_rowwise_sum",
                                   &BM_Eigen_reduce_rowwise_sum, stride, size,
                                   iter);

      benchmark::RegisterBenchmark("BM_ATen_reduce_sum", &BM_ATen_reduce_sum,
                                   stride, size, iter);
      benchmark::RegisterBenchmark("BM_ATen_reduce_colwise_sum",
                                   &BM_ATen_reduce_colwise_sum, stride, size,
                                   iter);
      benchmark::RegisterBenchmark("BM_ATen_reduce_rowwise_sum",
                                   &BM_ATen_reduce_rowwise_sum, stride, size,
                                   iter);

      benchmark::RegisterBenchmark("BM_Eigen_reduce_prod",
                                   &BM_Eigen_reduce_prod, stride, size, iter);
      benchmark::RegisterBenchmark("BM_Eigen_reduce_colwise_prod",
                                   &BM_Eigen_reduce_colwise_prod, stride, size,
                                   iter);
      benchmark::RegisterBenchmark("BM_Eigen_reduce_rowwise_prod",
                                   &BM_Eigen_reduce_rowwise_prod, stride, size,
                                   iter);

      benchmark::RegisterBenchmark("BM_ATen_reduce_prod", &BM_ATen_reduce_prod,
                                   stride, size, iter);
      benchmark::RegisterBenchmark("BM_ATen_reduce_colwise_prod",
                                   &BM_ATen_reduce_colwise_prod, stride, size,
                                   iter);
      benchmark::RegisterBenchmark("BM_ATen_reduce_rowwise_prod",
                                   &BM_ATen_reduce_rowwise_prod, stride, size,
                                   iter);
    }
  }
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
}

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
