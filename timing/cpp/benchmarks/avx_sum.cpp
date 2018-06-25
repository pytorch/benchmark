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
#include <cmath>
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
constexpr size_t _WIDTH = 16;
constexpr size_t _VSIZE = 8; // 8 floats - 256bits - one fetch has 1024 bits
constexpr size_t _ROW = 8;   // 4; // chunk of columns per tile
constexpr size_t _COL = 8;   // 4; // chunk of vector row per tile

using namespace tbb;

// HELPER FUNCTIONS

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

static inline size_t _divup(size_t x, size_t y) { return ((x + y - 1) / y); }

// SUMALL

// ONECORE

void sum_tree(float &sum, const float *arr, size_t start, size_t end) {
  register size_t k;
  float sarr[8];
  __m256 part_sum, tmp_sum[4], tmp_sum1, tmp_sum2, tmp_sum3, a[8];
  part_sum = _mm256_set1_ps(0);
  k = start;
  for (; k < end; k += 64) {
    for (size_t i = 0; i < 8; i++) {
      a[i] = _mm256_loadu_ps(arr + k + i * 8);
    }
    for (size_t i = 0; i < 8; i += 2) {
      tmp_sum[i / 2] = _mm256_add_ps(a[i], a[i + 1]);
    }
    tmp_sum1 = _mm256_add_ps(tmp_sum[0], tmp_sum[1]);
    tmp_sum2 = _mm256_add_ps(tmp_sum[2], tmp_sum[3]);
    tmp_sum3 = _mm256_add_ps(tmp_sum1, tmp_sum2);
    part_sum = _mm256_add_ps(part_sum, tmp_sum3);
  }
  _mm256_store_ps(sarr, part_sum);
  sum = sarr[0];
  for (int i = 1; i < 8; i++) {
    sum += sarr[i];
  }
}

void sum_simple(float &sum, const float *arr, size_t start, size_t end) {
  register size_t k;
  __m256 a[4]; // 128 bytes (two cache lines)
  a[0] = _mm256_set1_ps(0);
  a[1] = _mm256_set1_ps(0);
  a[2] = _mm256_set1_ps(0);
  a[3] = _mm256_set1_ps(0);
  k = start;
  for (; k < end; k += 32) {
    for (size_t i = 0; i < 4; i++) {
      a[i] = _mm256_add_ps(a[i], _mm256_loadu_ps(arr + k + i * 8));
    }
  }
  for (size_t i = 0; i < 4; i++) {
    float sarr[8];
    _mm256_store_ps(sarr, a[i]);
    for (int i = 0; i < 8; i++) {
      sum += sarr[i];
    }
  }
}

void sum_width(float &sum, const float *arr, size_t start, size_t end) {
  assert(end % _WIDTH == 0);
  assert(start % _WIDTH == 0);
  sum = 0;
  for (size_t i = start; i < end; i += _WIDTH) {
    float slocal = 0;
    for (size_t j = 0; j < _WIDTH; j++) {
      slocal += arr[i + j];
    }
    sum += slocal;
  }
}

void sum_naive(float &sum, const float *arr, size_t start, size_t end) {
  sum = 0;
  for (size_t i = start; i < end; i += 1) {
    sum += arr[i];
  }
}

// Simply way too slow
// void sum_std(float &sum, const float *arr, size_t start, size_t end) {
//  sum = std::accumulate(arr + start, arr + end, 0);
//}

// PARALLEL

class SumFoo {
  const float *my_a;

public:
  float my_sum;
  void operator()(const blocked_range<size_t> &r) {
    const float *a = my_a;
    float sum;
    sum_simple(sum, a, r.begin(), r.end());
    my_sum += sum;
  }

  SumFoo(SumFoo &x, split) : my_a(x.my_a), my_sum(0) {}

  void join(const SumFoo &y) { my_sum += y.my_sum; }

  SumFoo(const float *a) : my_a(a), my_sum(0) {}
};

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

void sum_omp_64_sum_simple(float &sum, const float *a, size_t start, size_t end,
                       size_t threshold, size_t max_num_thread) {
  (void)max_num_thread;
  (void)threshold;
#pragma omp parallel for reduction(+ : sum)
  for (size_t i = start; i < end; i += 64) {
    float sum_l = 0;
    sum_simple(sum_l, a, i, i + 64);
    sum += sum_l;
  }
}

void sum_omp_threshold_sum_simple(float &sum, const float *a, size_t start,
                              size_t end, size_t threshold,
                              size_t max_num_thread) {
  (void)max_num_thread;
  if (start + threshold > end) {
    sum_simple(sum, a, start, end);
  } else {
#pragma omp parallel for reduction(+ : sum)
    for (size_t i = start; i < end; i += threshold) {
      float sum_l = 0;
      sum_simple(sum_l, a, i, i + threshold);
      sum += sum_l;
    }
  }
}

void sum_tbb_for_sum_simple(float &sum, const float *a, size_t start, size_t end,
                        size_t threshold, size_t max_num_thread) {
  (void)max_num_thread;
  if (threshold > (end - start))
    threshold = (end - start);
  float *result = (float *)malloc(sizeof(float) * (end - start) / threshold);
  for (size_t i = 0; i < (end - start) / threshold; i++) {
    result[i] = 0;
  }
  tbb::parallel_for(blocked_range<size_t>(start, end, threshold),
                    [=](const blocked_range<size_t> &r) {
                      float sum_l;
                      sum_simple(sum_l, a, r.begin(), r.end());
                      result[r.begin() / threshold] = sum_l;
                    });
  for (size_t i = 0; i < (end - start) / threshold; i++) {
    sum += result[i];
  }
}

void sum_tbb_sum_naive(float &sum, const float *a, size_t start,
                            size_t end, size_t threshold,
                            size_t max_num_thread) {
  (void)max_num_thread;
  if (threshold > (end - start))
    threshold = (end - start);
  float *result = (float *)malloc(sizeof(float) * (end - start) / threshold);
  for (size_t i = 0; i < (end - start) / threshold; i++) {
    result[i] = 0;
  }
  tbb::parallel_for(blocked_range<size_t>(start, end, threshold),
                    [=](const blocked_range<size_t> &r) {
                      result[r.begin() / threshold] = 0;
                      for (size_t i = r.begin(); i != r.end(); ++i)
                        result[r.begin() / threshold] += a[i];
                    });
  for (size_t i = 0; i < (end - start) / threshold; i++) {
    sum += result[i];
  }
}

void sum_tbb_ap_arena(float &sum, const float *a, size_t start, size_t end,
                      size_t threshold, size_t max_num_thread) {
  static std::map<int64_t, tbb::task_arena> arenas = {
      {1, tbb::task_arena(1)},   {2, tbb::task_arena(2)},
      {4, tbb::task_arena(4)},   {8, tbb::task_arena(8)},
      {16, tbb::task_arena(16)}, {32, tbb::task_arena(32)}};
  if (end - start < threshold) {
    sum_simple(sum, a, start, end);
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
    sum = sf.my_sum;
  }
}

void sum_tbb_simp(float &sum, const float *a, size_t start, size_t end,
                  size_t threshold, size_t max_num_thread) {
  (void)max_num_thread;
  if (end - start < threshold) {
    sum_simple(sum, a, start, end);
  } else {
    SumFoo sf(a);
    static simple_partitioner ap;
    parallel_reduce(blocked_range<size_t>(start, end, threshold), sf, ap);
    sum = sf.my_sum;
  }
}

void sum_tbb_ap(float &sum, const float *a, size_t start, size_t end,
                size_t threshold, size_t max_num_thread) {
  (void)max_num_thread;
  if (end - start < threshold) {
    sum_simple(sum, a, start, end);
  } else {
    SumFoo sf(a);
    static affinity_partitioner ap;
    parallel_reduce(blocked_range<size_t>(start, end, threshold), sf, ap);
    sum = sf.my_sum;
  }
}

void sum_tbb_default(float &sum, const float *a, size_t start, size_t end,
                     size_t threshold, size_t max_num_thread) {
  (void)max_num_thread;
  if (end - start < threshold) {
    sum_simple(sum, a, start, end);
  } else {
    SumFoo sf(a);
    parallel_reduce(blocked_range<size_t>(start, end, threshold), sf);
    sum = sf.my_sum;
  }
}

// REDUCESUM

// ONECORE

void reducesum_impl_naive(const float *arr, float *outarr, size_t size1b,
                          size_t size1e, size_t size2b, size_t size2e,
                          size_t size2) {
  for (size_t i = size1b; i < size1e; i += 1) {
    for (size_t j = size2b; j < size2e; j += 1) {
      outarr[j] += arr[i * size2 + j];
    }
  }
}

void reducesum_impl_tile(const float *arr, float *outarr, size_t size1b,
                         size_t size1e, size_t size2b, size_t size2e,
                         size_t size2) {

  for (size_t i = size1b; i < size1e; i += _ROW) {
    for (size_t j = size2b / _VSIZE; j < size2e / _VSIZE; j += _COL) {
      __m256 tmp1[_COL];
      for (size_t j1 = 0; j1 < _COL; j1++) {
        __m256 tmp2[_ROW];
        tmp1[j1] = _mm256_load_ps(outarr + (j + j1) * _VSIZE);
        for (size_t i1 = 0; i1 < _ROW; i1++) {
          tmp2[i1] = _mm256_load_ps(arr + (i + i1) * size2 + (j + j1) * _VSIZE);
        }
        for (size_t i1 = 0; i1 < _ROW; i1++) {
          tmp1[j1] = _mm256_add_ps(tmp1[j1], tmp2[i1]);
        }
      }
      for (size_t j1 = 0; j1 < _COL; j1++) {
        _mm256_store_ps(outarr + (j + j1) * _VSIZE, tmp1[j1]);
      }
    }
  }
}

void reducesum_impl_simple(const float *arr, float *outarr, size_t size1b,
                           size_t size1e, size_t size2b, size_t size2e,
                           size_t size2) {
  assert(size2e > 7);
  size_t end = size2e - 7;
  for (size_t k = size2b; k < end; k += 8) {
    __m256 b = _mm256_loadu_ps(outarr + k);
    for (size_t i = size1b; i < size1e; i++) {
      __m256 a = _mm256_loadu_ps(arr + i * size2 + k);
      b = _mm256_add_ps(a, b);
    }
    _mm256_storeu_ps(outarr + k, b);
  }
}

void reducesum_impl_64(const float *arr, float *outarr, size_t size1b,
                       size_t size1e, size_t size2b, size_t size2e,
                       size_t size2) {
  assert(size2e > 63);
  size_t end = _divup(size2e, 64) * 64;
  for (size_t k = size2b; k < end; k += 64) {
    __m256 a[8];
    __m256 b[8];
    for (int ib = 0; ib < 8; ib++) {
      b[ib] = _mm256_loadu_ps(outarr + k + ib * 8);
    }
    for (size_t i = size1b; i < size1e; i += 1) {
      for (int ib = 0; ib < 8; ib++) {
        a[ib] = _mm256_loadu_ps(arr + i * size2 + k + ib * 8);
        b[ib] = _mm256_add_ps(a[ib], b[ib]);
      }
    }
    for (int ib = 0; ib < 8; ib++) {
      _mm256_storeu_ps(outarr + k + ib * 8, b[ib]);
    }
  }
}

// PARALLEL

class ReduceSumFoo {
  const float *my_a;
  size_t my_size2;

public:
  std::vector<float> my_sum;
  void operator()(const blocked_range2d<size_t> &r) {
    const float *a = my_a;
    float *sum = my_sum.data();
    size_t size2 = my_size2;
    reducesum_impl_64(a, sum, r.rows().begin(), r.rows().end(),
                      r.cols().begin(), r.cols().end(), size2);
  }

  ReduceSumFoo(ReduceSumFoo &x, split)
      : my_a(x.my_a), my_size2(x.my_size2), my_sum(x.my_sum) {}

  void join(const ReduceSumFoo &y) {
    float *sum1 = my_sum.data();
    const float *sum2 = y.my_sum.data();
    register size_t k;
    size_t end = my_size2 > 7 ? my_size2 - 8 : 0;
    k = 0;
    for (; k < end; k += 8) {
      register __m256 a, b;
      a = _mm256_loadu_ps(sum1 + k);
      b = _mm256_loadu_ps(sum2 + k);
      a = _mm256_add_ps(a, b);
      _mm256_storeu_ps(sum1 + k, a);
    }
    for (; k < my_size2; k += 1) {
      sum1[k] += sum2[k];
    }
  }

  ReduceSumFoo(const float a[], size_t size2)
      : my_a(a), my_size2(size2), my_sum(size2, 0.0f) {}
};

void reducesum_tbb(const float *arr, float *outarr, size_t size1b,
                        size_t size1e, size_t size2b, size_t size2e,
                        size_t size2, size_t threshold, size_t num_thread) {
  (void)num_thread;
  (void)threshold;
  ReduceSumFoo sf(arr, size2);
  parallel_reduce(
      blocked_range2d<size_t>(size1b, size1e, 64, size2b, size2e, 64), sf);
  for (size_t i = 0; i < size2; i++) {
    outarr[i] = sf.my_sum[i];
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

int main(int argc, char **argv) {
  std::map<std::string, void (*)(float &, const float *, size_t, size_t)>
      sum_funcs;

  sum_funcs["sum_naive"] = &sum_naive;
  sum_funcs["sum_simple"] = &sum_simple;
  sum_funcs["sum_tree"] = &sum_tree;
  sum_funcs["sum_width"] = &sum_width;

  std::map<std::string, void (*)(const float *, float *, size_t, size_t, size_t,
                                 size_t, size_t)>
      reducesum_funcs;

  reducesum_funcs["reducesum_impl_64"] = &reducesum_impl_64;
  reducesum_funcs["reducesum_impl_naive"] = &reducesum_impl_naive;
  reducesum_funcs["reducesum_impl_simple"] = &reducesum_impl_simple;
  reducesum_funcs["reducesum_impl_tile"] = &reducesum_impl_tile;


  std::map<std::string,
           void (*)(float &, const float *, size_t, size_t, size_t, size_t)>
      parallelsum_funcs;

  parallelsum_funcs["sum_omp_naive_simd"] = &sum_omp_naive_simd;
  parallelsum_funcs["sum_omp_naive"] = &sum_omp_naive;
  parallelsum_funcs["sum_omp_64_sum_simple"] = &sum_omp_64_sum_simple;
  parallelsum_funcs["sum_omp_threshold_sum_simple"] = &sum_omp_threshold_sum_simple;
  parallelsum_funcs["sum_tbb_for_sum_simple"] = &sum_tbb_for_sum_simple;
  parallelsum_funcs["sum_tbb_sum_naive"] = &sum_tbb_sum_naive;
  parallelsum_funcs["sum_tbb_ap_arena"] = &sum_tbb_ap_arena;
  parallelsum_funcs["sum_tbb_simp"] = &sum_tbb_simp;
  parallelsum_funcs["sum_tbb_ap"] = &sum_tbb_ap;
  parallelsum_funcs["sum_tbb_default"] = &sum_tbb_default;

  std::map<std::string, void (*)(const float *, float *, size_t, size_t, size_t,
                                 size_t, size_t, size_t, size_t)>
      parallelreducesum_funcs;

  parallelreducesum_funcs["reducesum_tbb"] = &reducesum_tbb;

  int64_t min_s = 8 << 12;
  int64_t max_s = 8 << 24;
  int64_t ratio_s = max_s / min_s;
  int64_t min_th = 8 * 1024;
  int64_t max_th = 128 * 1024;


  int64_t min_nt = 2;
  int64_t max_nt = 20;

  for (int64_t s = min_s; s < max_s; s *= 4) {
    for (auto &kv : sum_funcs) {
      benchmark::RegisterBenchmark(kv.first.c_str(), &BM_ONECORE_SUM, s, 128,
                                   kv.second);
    }
  }

  for (int64_t k = 2; k < ratio_s; k = k * 2) {
    int64_t so = max_s / (ratio_s  * k);
    int64_t si = ratio_s * k;
    for (auto &kv : reducesum_funcs) {
      benchmark::RegisterBenchmark(kv.first.c_str(), &BM_ONECORE_REDUCESUM, so,
                                   si, 16, kv.second);
    }
  }

  for (int64_t nt = min_nt; nt < max_nt; nt *= 4) {
    for (int64_t s = min_s; s < max_s; s *= 4) {
      for (int64_t th = min_th; th < max_th; th *= 4) {
        for (auto &kv : parallelsum_funcs) {
          benchmark::RegisterBenchmark(kv.first.c_str(), &BM_PARALLEL_SUM, s,
                                       128, th, nt, kv.second);
        }
      }
    }
    for (int64_t k = 2; k < ratio_s; k = k * 2) {
      int64_t so = max_s / (ratio_s * k);
      int64_t si = ratio_s * k;
      for (int64_t th = min_th; th < max_th; th *= 4) {
        for (auto &kv : parallelreducesum_funcs) {
          benchmark::RegisterBenchmark(kv.first.c_str(), &BM_PARALLEL_REDUCESUM,
                                       so, si, 128, th, nt, kv.second);
        }
      }
    }
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
}
