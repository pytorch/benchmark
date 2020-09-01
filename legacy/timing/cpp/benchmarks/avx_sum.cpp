#include "immintrin.h"
#include "tbb/blocked_range.h"
#include "tbb/parallel_reduce.h"
#include "tbb/partitioner.h"
#include "tbb/task_arena.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/tbb.h"
#include "tbb/tick_count.h"
#include "xmmintrin.h"
#include <benchmark/benchmark.h>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <omp.h>
#include <random>
#include <stdexcept>
#include <vector>

// Mimic TH alignment
constexpr size_t _ALIGNMENT = 64;

using namespace tbb;

// HELPER FUNCTIONS

void make_vector(float *data_, size_t size) {
  for (size_t i = 0; i < size; i++) {
    data_[i] = (float)(i % 1024);
  }
}

static inline int64_t round_down(int64_t a, int64_t m) { return a - (a % m); }

inline int64_t divup(int64_t x, int64_t y) { return (x + y - 1) / y; }

void make_random_vector(float *data_, size_t size) {
  std::random_device
      rd; // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::normal_distribution<> dis{0, 1};
  for (size_t i = 0; i < size; i++) {
    data_[i] = (float)(dis(gen));
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

static inline size_t _divup(size_t x, size_t y) { return ((x + y - 1) / y); }

// SUMALL

// ONECORE

inline void sum_naive(float &sum, const float *arr, size_t start, size_t end) {
  for (size_t i = start; i < end; i += 1) {
    sum += arr[i];
  }
}

inline void sum_naive_32(float &sum, const float *arr, size_t start,
                         size_t end) {
  int64_t blocks = (end - start) / 32;
  for (int64_t k = 0; k < blocks; k++) {
    float slocal = 0;
    for (size_t j = 0; j < 32; j++) {
      slocal += arr[start + k * 32 + j];
    }
    sum += slocal;
  }
  sum_naive(sum, arr, start + blocks * 32, end);
}

inline void sum_simple_128(float &sum, const float *arr, size_t start,
                           size_t end) {
  __m256 a[4]; // 128 bytes (two cache lines)
  a[0] = _mm256_set1_ps(0);
  a[1] = _mm256_set1_ps(0);
  a[2] = _mm256_set1_ps(0);
  a[3] = _mm256_set1_ps(0);
  for (int64_t k = 0; k < (end - start) / 32; k++) {
    for (size_t i = 0; i < 4; i++) {
      a[i] = _mm256_add_ps(a[i], _mm256_loadu_ps(arr + start + k * 32 + i * 8));
    }
  }
  for (size_t i = 0; i < 4; i++) {
    float sarr[8];
    _mm256_store_ps(sarr, a[i]);
    for (int i = 0; i < 8; i++) {
      sum += sarr[i];
    }
  }
  sum_naive(sum, arr, start + ((end - start) / 32) * 32, end);
}

inline void sum_simple_128_aligned(float &sum, const float *arr, size_t start,
                                   size_t end) {
  __m256 a[4]; // 128 bytes (two cache lines)
  a[0] = _mm256_set1_ps(0);
  a[1] = _mm256_set1_ps(0);
  a[2] = _mm256_set1_ps(0);
  a[3] = _mm256_set1_ps(0);
  for (int64_t k = 0; k < (end - start) / 32; k++) {
    for (size_t i = 0; i < 4; i++) {
      a[i] = _mm256_add_ps(a[i], _mm256_load_ps(arr + start + k * 32 + i * 8));
    }
  }
  for (size_t i = 0; i < 4; i++) {
    float sarr[8];
    _mm256_store_ps(sarr, a[i]);
    for (int i = 0; i < 8; i++) {
      sum += sarr[i];
    }
  }
  sum_naive(sum, arr, start + ((end - start) / 32) * 32, end);
}

inline void sum_simple_256(float &sum, const float *arr, size_t start,
                           size_t end) {
  __m256 a[8]; // 128 bytes (two cache lines)
  a[0] = _mm256_set1_ps(0);
  a[1] = _mm256_set1_ps(0);
  a[2] = _mm256_set1_ps(0);
  a[3] = _mm256_set1_ps(0);
  a[4] = _mm256_set1_ps(0);
  a[5] = _mm256_set1_ps(0);
  a[6] = _mm256_set1_ps(0);
  a[7] = _mm256_set1_ps(0);
  for (int64_t k = 0; k < (end - start) / 64; k++) {
    for (size_t i = 0; i < 8; i++) {
      a[i] = _mm256_add_ps(a[i], _mm256_loadu_ps(arr + start + k * 64 + i * 8));
    }
  }
  for (size_t i = 0; i < 8; i++) {
    float sarr[8];
    _mm256_store_ps(sarr, a[i]);
    for (int i = 0; i < 8; i++) {
      sum += sarr[i];
    }
  }
  sum_naive(sum, arr, start + ((end - start) / 64) * 64, end);
}

inline void sum_simple(float &sum, const float *arr, size_t start, size_t end) {
  register size_t k;
  __m256 a; // 128 bytes (two cache lines)
  a = _mm256_set1_ps(0);
  int64_t blocks = (end - start) / 8;
  for (int64_t k = 0; k < blocks; k++) {
    a = _mm256_add_ps(a, _mm256_loadu_ps(arr + start + k * 8));
  }
  float sarr[8];
  _mm256_store_ps(sarr, a);
  for (int i = 0; i < 8; i++) {
    sum += sarr[i];
  }
  sum_naive(sum, arr, start + blocks * 8, end);
}

// Simply way too slow
// void sum_std(float &sum, const float *arr, size_t start, size_t end) {
//  sum = std::accumulate(arr + start, arr + end, 0);
//}

// PARALLEL

void sum_omp_naive_simd(float &sum, const float *a, size_t start, size_t end,
                        size_t threshold, size_t max_num_thread) {
  (void)max_num_thread;
  (void)threshold;
#pragma omp parallel for simd reduction(+ : sum)
  for (size_t i = start; i < end; i++) {
    sum += a[i];
  }
}

void sum_omp_naive(float &sum, const float *a, size_t start, size_t end,
                   size_t threshold, size_t max_num_thread) {
  (void)max_num_thread;
  (void)threshold;
#pragma omp parallel for reduction(+ : sum)
  for (size_t i = start; i < end; i++) {
    sum += a[i];
  }
}

void sum_omp_reduce_128(float &sum, const float *a, size_t start, size_t end,
                        size_t threshold, size_t max_num_thread) {
#pragma omp parallel for reduction(+ : sum)
  for (size_t i = start; i < end; i += threshold) {
    float result = 0;
    sum_simple_128(result, a, i, std::min(i + threshold, end));
    sum += result;
  }
}

void sum_omp_simple_128(float &sum, const float *a, size_t start_, size_t end_,
                        size_t threshold, size_t max_num_thread) {
  (void)max_num_thread;
  int64_t num_threads = omp_get_max_threads();
  std::vector<float> results(num_threads, 0);
  float *results_data = results.data();
  int64_t end = end_;
  int64_t start = start_;
  int64_t range = end - start;
#pragma omp parallel if ((end - start) > threshold)
  {
    int64_t num_threads = omp_get_num_threads();
    int64_t tid = omp_get_thread_num();
    int64_t chunk_size = divup(range, num_threads);
    int64_t start_tid = start + tid * chunk_size;
    float result = 0;
    sum_simple_128(result, a, start_tid, std::min(end, chunk_size + start_tid));
    results_data[tid] = result;
  }
  for (int64_t i = 0; i < num_threads; i++) {
    sum += results[i];
  }
}

class SumFoo {
  const float *my_a;

public:
  float my_sum;
  void operator()(const blocked_range<size_t> &r) {
    const float *a = my_a;
    float sum = 0;
    sum_simple_128(sum, a, r.begin(), r.end());
    my_sum += sum;
  }
  SumFoo(SumFoo &x, split) : my_a(x.my_a), my_sum(0) {}
  void join(const SumFoo &y) { my_sum += y.my_sum; }
  SumFoo(const float *a) : my_a(a), my_sum(0) {}
};

void sum_tbb_ap_arena(float &sum, const float *a, size_t start, size_t end,
                      size_t threshold, size_t max_num_thread) {
  static std::map<int64_t, tbb::task_arena> arenas = {
      {1, tbb::task_arena(1)},   {2, tbb::task_arena(2)},
      {4, tbb::task_arena(4)},   {8, tbb::task_arena(8)},
      {16, tbb::task_arena(16)}, {32, tbb::task_arena(32)}};
  if (end - start < threshold) {
    sum_simple_128(sum, a, start, end);
  } else {
    size_t max_tasks = ((end - start) / threshold);
    SumFoo sf(a);
    static affinity_partitioner ap;
    if (max_tasks < max_num_thread) {
      if (arenas.count(max_tasks) == 0) {
        std::cout << "need arena for " << max_tasks << std::endl;
      }
      arenas[max_tasks].execute([&] {
        parallel_reduce(blocked_range<size_t>(start, end, threshold), sf, ap);
      });
    } else {
      parallel_reduce(blocked_range<size_t>(start, end, threshold), sf, ap);
    }
    sum += sf.my_sum;
  }
}

void sum_tbb_simp(float &sum, const float *a, size_t start, size_t end,
                  size_t threshold, size_t max_num_thread) {
  (void)max_num_thread;
  static simple_partitioner ap;
  sum += parallel_reduce(
      blocked_range<size_t>(start, end, threshold), 0.f,
      [a](const tbb::blocked_range<size_t> &r, float init) -> float {
        float result = init;
        sum_simple_128(result, a, r.begin(), r.end());
        return result;
      },
      std::plus<float>(), ap);
}

void sum_tbb_ap(float &sum, const float *a, size_t start, size_t end,
                size_t threshold, size_t max_num_thread) {
  (void)max_num_thread;
  static affinity_partitioner ap;
  sum += parallel_reduce(
      blocked_range<size_t>(start, end, threshold), 0.f,
      [a](const tbb::blocked_range<size_t> &r, float init) -> float {
        float result = init;
        sum_simple_128(result, a, r.begin(), r.end());
        return result;
      },
      std::plus<float>(), ap);
}

void sum_tbb_default(float &sum, const float *a, size_t start, size_t end,
                     size_t threshold, size_t max_num_thread) {
  (void)max_num_thread;
  sum += parallel_reduce(
      blocked_range<int64_t>(start, end, threshold), 0.f,
      [a](const tbb::blocked_range<int64_t> &r, float init) -> float {
        float result = init;
        sum_simple_128(result, a, r.begin(), r.end());
        return result;
      },
      std::plus<float>());
}

// REDUCESUM

// ONECORE

void reducesum_naive(const float *arr, float *outarr, size_t size1b,
                     size_t size1e, size_t size2b, size_t size2e,
                     size_t size2) {
  for (size_t i = size1b; i < size1e; i += 1) {
    for (size_t j = size2b; j < size2e; j += 1) {
      outarr[j] += arr[i * size2 + j];
    }
  }
}

void reducesum_simple(const float *arr, float *outarr, size_t size1b,
                      size_t size1e, size_t size2b, size_t size2e,
                      size_t size2) {
  int64_t blocks2 = (size2e - size2b) / 8;
  for (size_t k = 0; k < blocks2; k++) {
    __m256 b = _mm256_loadu_ps(outarr + size2b + k * 8);
    for (size_t i = size1b; i < size1e; i++) {
      __m256 a = _mm256_loadu_ps(arr + i * size2 + size2b + k * 8);
      b = _mm256_add_ps(a, b);
    }
    _mm256_storeu_ps(outarr + size2b + k * 8, b);
  }
  for (size_t j = size2b + blocks2 * 8; j < size2e; j += 1) {
    for (size_t i = size1b; i < size1e; i += 1) {
      outarr[j] += arr[i * size2 + j];
    }
  }
}

void reducesum_simple_128(const float *arr, float *outarr, size_t size1b,
                          size_t size1e, size_t size2b, size_t size2e,
                          size_t size2) {
  size_t blocks2 = (size2e - size2b) / 32;
  for (size_t k = 0; k < blocks2; k++) {
    __m256 b[4];
    for (size_t ib = 0; ib < 4; ib++) {
      b[ib] = _mm256_loadu_ps(outarr + size2b + k * 32 + ib * 8);
    }
    for (size_t i = size1b; i < size1e; i += 1) {
      for (size_t ib = 0; ib < 4; ib++) {
        __m256 val =
            _mm256_loadu_ps(arr + i * size2 + size2b + k * 32 + ib * 8);
        b[ib] = _mm256_add_ps(val, b[ib]);
      }
    }
    for (size_t ib = 0; ib < 4; ib++) {
      _mm256_storeu_ps(outarr + size2b + k * 32 + ib * 8, b[ib]);
    }
  }
  for (size_t j = size2b + blocks2 * 32; j < size2e; j += 1) {
    for (size_t i = size1b; i < size1e; i += 1) {
      outarr[j] += arr[i * size2 + j];
    }
  }
}

// PARALLEL

void reducesum_omp_simple_128(const float *arr, float *outarr, size_t size1b,
                              size_t size1e, size_t size2b, size_t size2e,
                              size_t size2, size_t threshold,
                              size_t num_thread) {
  (void)num_thread;
  (void)threshold;
#pragma omp parallel for
  for (size_t i = size2b; i < size2e; i += threshold) {
    reducesum_simple_128(arr, outarr, size1b, size1e, i,
                         std::min(i + threshold, size2e), size2);
  }
}

void reducesum_tbb_simple_128(const float *arr, float *outarr, size_t size1b,
                              size_t size1e, size_t size2b, size_t size2e,
                              size_t size2, size_t threshold,
                              size_t num_thread) {
  (void)num_thread;
  static affinity_partitioner ap;
  parallel_for(blocked_range<size_t>(size2b, size2e, threshold),
               [&](const tbb::blocked_range<size_t> &r) {
                 reducesum_simple_128(arr, outarr, size1b, size1e, r.begin(),
                                      r.end(), size2);
               },
               ap);
}

void reducesum_tbb_simple_128_arena(const float *arr, float *outarr,
                                    size_t size1b, size_t size1e, size_t size2b,
                                    size_t size2e, size_t size2,
                                    size_t threshold, size_t max_num_thread) {
  static std::map<int64_t, tbb::task_arena> arenas = {
      {1, tbb::task_arena(1)},   {2, tbb::task_arena(2)},
      {4, tbb::task_arena(4)},   {8, tbb::task_arena(8)},
      {16, tbb::task_arena(16)}, {32, tbb::task_arena(32)}};
  if (((size2e - size2b)) < threshold) {
    reducesum_simple_128(arr, outarr, size1b, size1e, size2b, size2e, size2);
  } else {
    size_t max_tasks = ((size2e - size2b) / threshold);
    static affinity_partitioner ap;
    if (max_tasks < max_num_thread) {
      if (arenas.count(max_tasks) == 0) {
        std::cout << "need arena for " << max_tasks << std::endl;
      }
      arenas[max_tasks].execute([&] {
        parallel_for(blocked_range<size_t>(size2b, size2e, threshold),
                     [&](const tbb::blocked_range<size_t> &r) {
                       reducesum_simple_128(arr, outarr, size1b, size1e,
                                            r.begin(), r.end(), size2);
                     },
                     ap);
      });
    } else {
      parallel_for(blocked_range<size_t>(size2b, size2e, threshold),
                   [&](const tbb::blocked_range<size_t> &r) {
                     reducesum_simple_128(arr, outarr, size1b, size1e,
                                          r.begin(), r.end(), size2);
                   },
                   ap);
    }
  }
}

static void BM_ONECORE_SUM(benchmark::State &state, int64_t size, int64_t iter,
                           void (*sumf)(float &, const float *, size_t,
                                        size_t)) {
  for (auto _ : state) {
    state.PauseTiming();
    state.counters["iter"] = iter;
    state.counters["num_thread"] = -1;
    state.counters["size"] = size;
    state.counters["size_inner"] = -1;
    state.counters["size_outer"] = -1;
    state.counters["threshold"] = -1;
    int64_t steps = iter;
    float sum = get_random_value();
    float *data_ = NULL;
    make_float_data(&data_, size);
    make_vector(data_, size);
    state.ResumeTiming();
    for (int64_t step = 0; step < iter; step++) {
      sumf(sum, data_, 0, size);
    }
    state.PauseTiming();
    free(data_);
    state.ResumeTiming();
  }
}

static void BM_ONECORE_REDUCESUM(benchmark::State &state, int64_t size_outer,
                                 int64_t size_inner, int64_t iter,
                                 void (*reducesumf)(const float *, float *,
                                                    size_t, size_t, size_t,
                                                    size_t, size_t)) {
  for (auto _ : state) {
    state.PauseTiming();
    state.counters["iter"] = iter;
    state.counters["num_thread"] = -1;
    state.counters["size"] = -1;
    state.counters["size_inner"] = size_inner;
    state.counters["size_outer"] = size_outer;
    state.counters["threshold"] = -1;
    int64_t steps = iter;
    float sum = get_random_value();
    float *data_ = NULL;
    make_float_data(&data_, size_outer * size_inner);
    make_vector(data_, size_outer * size_inner);
    float *out_data_ = NULL;
    make_float_data(&out_data_, size_inner);
    make_vector(out_data_, size_inner);
    state.ResumeTiming();
    for (int64_t step = 0; step < iter; step++) {
      reducesumf(data_, out_data_, 0, size_outer, 0, size_inner, size_inner);
    }
    state.PauseTiming();
    free(data_);
    free(out_data_);
    state.ResumeTiming();
  }
}

static void BM_PARALLEL_SUM(benchmark::State &state, int64_t size, int64_t iter,
                            int64_t threshold, int64_t num_thread,
                            void (*psumf)(float &, const float *, size_t,
                                          size_t, size_t, size_t)) {
  for (auto _ : state) {
    state.PauseTiming();
    state.counters["iter"] = iter;
    state.counters["num_thread"] = num_thread;
    state.counters["size"] = size;
    state.counters["size_inner"] = -1;
    state.counters["size_outer"] = -1;
    state.counters["threshold"] = threshold;
    int64_t steps = iter;
    float sum = get_random_value();
    float *data_ = NULL;
    make_float_data(&data_, size);
    make_vector(data_, size);
    task_scheduler_init init(num_thread);
    omp_set_num_threads(num_thread);
    state.ResumeTiming();
    for (int64_t step = 0; step < iter; step++) {
      psumf(sum, data_, 0, size, threshold, num_thread);
    }
    state.PauseTiming();
    init.terminate();
    free(data_);
    state.ResumeTiming();
  }
}

static void BM_PARALLEL_REDUCESUM(
    benchmark::State &state, int64_t size_outer, int64_t size_inner,
    int64_t iter, int64_t threshold, int64_t num_thread,
    void (*preducesumf)(const float *, float *, size_t, size_t, size_t, size_t,
                        size_t, size_t, size_t)) {
  for (auto _ : state) {
    state.PauseTiming();
    state.counters["iter"] = iter;
    state.counters["num_thread"] = num_thread;
    state.counters["size"] = -1;
    state.counters["size_inner"] = size_inner;
    state.counters["size_outer"] = size_outer;
    state.counters["threshold"] = threshold;
    int64_t steps = iter;
    float sum = get_random_value();
    float *data_ = NULL;
    make_float_data(&data_, size_outer * size_inner);
    make_vector(data_, size_outer * size_inner);
    float *out_data_ = NULL;
    make_float_data(&out_data_, size_inner);
    make_vector(out_data_, size_inner);
    task_scheduler_init init(num_thread);
    omp_set_num_threads(num_thread);
    state.ResumeTiming();
    for (int64_t step = 0; step < iter; step++) {
      preducesumf(data_, out_data_, 0, size_outer, 0, size_inner, size_inner,
                  threshold, num_thread);
    }
    state.PauseTiming();
    init.terminate();
    free(data_);
    free(out_data_);
    state.ResumeTiming();
  }
}

void test_sum(std::string name,
              void (*sumf_comp)(float &, const float *, size_t, size_t)) {
  int64_t size = 367 * 107;
  float *data_ = NULL;
  make_float_data(&data_, size);
  make_random_vector(data_, size);
  task_scheduler_init init(10);
  omp_set_num_threads(10);
  for (int64_t offset = 0; offset < 1000; offset = (offset + 3) * 13) {
    float sum_ref = 0;
    float sum_comp = 0;
    sum_naive(sum_ref, data_, offset, size - offset);
    sumf_comp(sum_comp, data_, offset, size - offset);
    float ratio = std::abs(sum_ref - sum_comp) / std::abs(sum_ref);
    if (ratio > 1e-3) {
      for (int64_t i = offset; i < size - offset; i++) {
        std::cout << "i: " << i << " - " << data_[i] << std::endl;
      }
      throw std::runtime_error("test_sum failed - name: " + name +
                               " - sum_ref: " + std::to_string(sum_ref) +
                               " - sum_comp: " + std::to_string(sum_comp) +
                               " - error: " + std::to_string(ratio));
    }
  }
  init.terminate();
  free(data_);
}

void test_sum_parallel(std::string name,
                       void (*sumf_comp)(float &, const float *, size_t, size_t,
                                         size_t, size_t)) {
  int64_t size = 367 * 107 * 10;
  float *data_ = NULL;
  // It is very important that his happens here!
  // Otherwise subsequent calls to tbb parallel constructs won't be affected
  // by the number of threads set if this is called first.
  task_scheduler_init init(10);
  omp_set_num_threads(10);
  make_float_data(&data_, size);
  make_random_vector(data_, size);
  for (int64_t offset = 0; offset < 1000; offset = (offset + 3) * 13) {
    float sum_ref = 0;
    float sum_comp = 0;
    sum_naive(sum_ref, data_, offset, size - offset);
    sumf_comp(sum_comp, data_, offset, size - offset, 128,
              omp_get_max_threads());
    float ratio = std::abs(sum_ref - sum_comp) / std::abs(sum_ref);
    if (ratio > 1e-3) {
      throw std::runtime_error("test_sum_parallel failed - name: " + name +
                               " - sum_ref: " + std::to_string(sum_ref) +
                               " - sum_comp: " + std::to_string(sum_comp) +
                               " - error: " + std::to_string(ratio));
    }
  }
  init.terminate();
  free(data_);
}

void test_reducesum(std::string name,
                    void (*reducef_comp)(const float *, float *, size_t, size_t,
                                         size_t, size_t, size_t)) {

  size_t inner_size = 3670;
  size_t outer_size = 107 * 10;
  float *data_ = NULL;
  float *out_data_ = NULL;
  float *out_data_comp_ = NULL;
  // It is very important that his happens here!
  // Otherwise subsequent calls to tbb parallel constructs won't be affected
  // by the number of threads set if this is called first.
  task_scheduler_init init(10);
  omp_set_num_threads(10);
  make_float_data(&data_, outer_size * inner_size);
  make_random_vector(data_, outer_size * inner_size);
  make_float_data(&out_data_, inner_size);
  make_float_data(&out_data_comp_, inner_size);
  for (int64_t offset = 0; offset < 1000; offset = (offset + 3) * 13) {
    reducesum_naive(data_, out_data_, offset, outer_size - offset, offset,
                    inner_size - offset, inner_size);
    reducef_comp(data_, out_data_comp_, offset, outer_size - offset, offset,
                 inner_size - offset, inner_size);
    for (int64_t i = offset; i < inner_size - offset; i++) {
      float ratio =
          std::abs(out_data_[i] - out_data_comp_[i]) / std::abs(out_data_[i]);
      if (ratio > 1e-3) {
        std::string wrong_out = std::to_string(out_data_[i]);
        std::string wrong_out_comp = std::to_string(out_data_comp_[i]);
        std::string s1 = "test_reducesum failed - name: " + name;
        std::string s2 = " - out_data_[" + std::to_string(i);
        std::string s3 = "]: " + wrong_out + " - out_data_comp_[";
        std::string s4 = std::to_string(i) + "]: " + wrong_out_comp;
        std::string s5 = " - error: " + std::to_string(ratio);
        throw std::runtime_error(s1 + s2 + s3 + s4 + s5);
      }
    }
  }
  init.terminate();
  free(data_);
  free(out_data_);
  free(out_data_comp_);
}

void test_parallelreducesum(std::string name,
                            void (*parallelreducef_comp)(const float *, float *,
                                                         size_t, size_t, size_t,
                                                         size_t, size_t, size_t,
                                                         size_t)) {

  size_t inner_size = 3670;
  size_t outer_size = 107 * 10;
  float *data_ = NULL;
  float *out_data_ = NULL;
  float *out_data_comp_ = NULL;
  // It is very important that his happens here!
  // Otherwise subsequent calls to tbb parallel constructs won't be affected
  // by the number of threads set if this is called first.
  task_scheduler_init init(10);
  omp_set_num_threads(10);
  make_float_data(&data_, outer_size * inner_size);
  make_random_vector(data_, outer_size * inner_size);
  make_float_data(&out_data_, inner_size);
  make_float_data(&out_data_comp_, inner_size);
  for (int64_t offset = 0; offset < 1000; offset = (offset + 3) * 13) {
    reducesum_naive(data_, out_data_, 0, outer_size, offset,
                    inner_size - offset, inner_size);
    parallelreducef_comp(data_, out_data_comp_, 0, outer_size, offset,
                         inner_size - offset, inner_size, 128,
                         omp_get_max_threads());
    for (int64_t i = offset; i < inner_size - offset; i++) {
      float ratio =
          std::abs(out_data_[i] - out_data_comp_[i]) / std::abs(out_data_[i]);
      if (ratio > 1e-3) {
        std::string wrong_out = std::to_string(out_data_[i]);
        std::string wrong_out_comp = std::to_string(out_data_comp_[i]);
        std::string s1 = "test_parallelreducesum failed - name: " + name;
        std::string s2 = " - out_data_[" + std::to_string(i);
        std::string s3 = "]: " + wrong_out + " - out_data_comp_[";
        std::string s4 = std::to_string(i) + "]: " + wrong_out_comp;
        std::string s5 = " - error: " + std::to_string(ratio);
        throw std::runtime_error(s1 + s2 + s3 + s4 + s5);
      }
    }
  }
  init.terminate();
  free(data_);
  free(out_data_);
  free(out_data_comp_);
}

int main(int argc, char **argv) {
  std::map<std::string, void (*)(float &, const float *, size_t, size_t)>
      sum_funcs;

  sum_funcs["sum_naive"] = &sum_naive;
  sum_funcs["sum_naive_32"] = &sum_naive_32;
  sum_funcs["sum_simple"] = &sum_simple;
  sum_funcs["sum_simple_128"] = &sum_simple_128;
  sum_funcs["sum_simple_128_aligned"] = &sum_simple_128_aligned;
  sum_funcs["sum_simple_256"] = &sum_simple_256;

  for (auto &kv : sum_funcs) {
    std::cerr << "Testing: " << kv.first << std::endl;
    test_sum(kv.first, kv.second);
  }

  std::map<std::string,
           void (*)(float &, const float *, size_t, size_t, size_t, size_t)>
      parallelsum_funcs;

  parallelsum_funcs["sum_omp_naive_simd"] = &sum_omp_naive_simd;
  parallelsum_funcs["sum_omp_naive"] = &sum_omp_naive;
  parallelsum_funcs["sum_omp_simple_128"] = &sum_omp_simple_128;
  parallelsum_funcs["sum_omp_reduce_128"] = &sum_omp_reduce_128;
  parallelsum_funcs["sum_tbb_simp"] = &sum_tbb_simp;
  parallelsum_funcs["sum_tbb_ap"] = &sum_tbb_ap;
  parallelsum_funcs["sum_tbb_ap_arena"] = &sum_tbb_ap_arena;
  parallelsum_funcs["sum_tbb_default"] = &sum_tbb_default;

  for (auto &kv : parallelsum_funcs) {
    std::cerr << "Testing: " << kv.first << std::endl;
    test_sum_parallel(kv.first, kv.second);
  }

  std::map<std::string, void (*)(const float *, float *, size_t, size_t, size_t,
                                 size_t, size_t)>
      reducesum_funcs;

  reducesum_funcs["reducesum_naive"] = &reducesum_naive;
  reducesum_funcs["reducesum_simple"] = &reducesum_simple;
  reducesum_funcs["reducesum_simple_128"] = &reducesum_simple_128;

  for (auto &kv : reducesum_funcs) {
    std::cerr << "Testing: " << kv.first << std::endl;
    test_reducesum(kv.first, kv.second);
  }

  std::map<std::string, void (*)(const float *, float *, size_t, size_t, size_t,
                                 size_t, size_t, size_t, size_t)>
      parallelreducesum_funcs;

  parallelreducesum_funcs["reducesum_omp_simple_128"] =
      &reducesum_omp_simple_128;
  parallelreducesum_funcs["reducesum_tbb_simple_128"] =
      &reducesum_tbb_simple_128;
  parallelreducesum_funcs["reducesum_tbb_simple_128_arena"] =
      &reducesum_tbb_simple_128_arena;

  for (auto &kv : parallelreducesum_funcs) {
    std::cerr << "Testing: " << kv.first << std::endl;
    test_parallelreducesum(kv.first, kv.second);
  }

  int64_t min_s = (8 << 12) / 2;
  int64_t max_s = 8 << 25;
  int64_t ratio_s = max_s / 2;
  int64_t min_th = 8 * 1024;
  int64_t max_th = 128 * 1024;

  int64_t min_nt = 2;
  int64_t max_nt = 20;

  for (int64_t s = min_s; s < max_s; s *= 2) {
    for (auto &kv : sum_funcs) {
      benchmark::RegisterBenchmark(kv.first.c_str(), &BM_ONECORE_SUM, s, 128,
                                   kv.second);
    }
  }

  for (int64_t kk = 1; kk < 8; kk = kk * 2) {
    for (int64_t k = 4; k < ratio_s / 4; k = k * 2) {
      int64_t so = max_s / k / kk / 16;
      int64_t si = k;
      if (so == 0 or si == 0) {
        continue;
      }
      for (auto &kv : reducesum_funcs) {
        benchmark::RegisterBenchmark(kv.first.c_str(), &BM_ONECORE_REDUCESUM,
                                     so, si, 16, kv.second);
      }
    }
  }

  for (int64_t nt = min_nt; nt < max_nt; nt *= 2) {
    for (int64_t s = min_s; s < max_s; s *= 4) {
      for (int64_t th = min_th; th < max_th; th *= 2) {
        for (auto &kv : parallelsum_funcs) {
          benchmark::RegisterBenchmark(kv.first.c_str(), &BM_PARALLEL_SUM, s,
                                       128, th, nt, kv.second);
        }
      }
    }
    for (int64_t kk = 1; kk < 8; kk = kk * 2) {
      for (int64_t k = 4; k < ratio_s / 4; k = k * 2) {
        int64_t so = max_s / k / kk / 16;
        int64_t si = k;
        if (so == 0 or si == 0) {
          continue;
        }
        for (int64_t th = min_th; th < max_th; th *= 4) {
          for (auto &kv : parallelreducesum_funcs) {
            benchmark::RegisterBenchmark(kv.first.c_str(),
                                         &BM_PARALLEL_REDUCESUM, so, si, 128,
                                         th, nt, kv.second);
          }
        }
      }
    }
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
}
