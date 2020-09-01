#include "tbb/blocked_range.h"
#include "tbb/parallel_reduce.h"
#include <benchmark/benchmark.h>
#include <omp.h>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>

float do_something(float r) {
    benchmark::DoNotOptimize(r = r * 2);
    return r;
}

float get_random_value() {
  std::random_device
      rd; // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<> dis(1, 6);
  return dis(gen);
}

#define SETTING ->RangeMultiplier(2)->Ranges({{8, 8 << 20}, {64, 128}});

static void BM_TBB_OMP(benchmark::State &state) {
  for (auto _ : state) {
    state.PauseTiming();
    state.counters["size"] = state.range(0);
    state.counters["iter"] = state.range(1);
    int64_t steps = state.range(0);
    float sum = get_random_value();
    state.ResumeTiming();
    for (int64_t step = 0; step < state.range(1); step++) {
#pragma omp parallel reduction(+ : sum)
      for (int64_t i = 0; i < steps; i++) {
        sum += i * sum;
      }
      sum = do_something(sum);
      sum += tbb::parallel_reduce(
          tbb::blocked_range<int64_t>(0, steps), 0.f,
          [](const tbb::blocked_range<int64_t> &r, float value) {
            for (int64_t i = r.begin(); i < r.end(); i++) {
              value += i * value;
            }
            return value;
          },
          std::plus<float>());
      sum = do_something(sum);
    }
  }
}

static void BM_TBB(benchmark::State &state) {
  for (auto _ : state) {
    state.PauseTiming();
    state.counters["size"] = state.range(0);
    state.counters["iter"] = state.range(1);
    int64_t steps = state.range(0);
    float sum = get_random_value();
    state.ResumeTiming();
    for (int64_t step = 0; step < state.range(1); step++) {
      sum += tbb::parallel_reduce(
          tbb::blocked_range<int64_t>(0, steps), 0.f,
          [](const tbb::blocked_range<int64_t> &r, float value) {
            for (int64_t i = r.begin(); i < r.end(); i++) {
              value += i * value;
            }
            return value;
          },
          std::plus<float>());
      sum = do_something(sum);
    }
  }
}

static void BM_OMP(benchmark::State &state) {
  for (auto _ : state) {
    state.PauseTiming();
    state.counters["size"] = state.range(0);
    state.counters["iter"] = state.range(1);
    int64_t steps = state.range(0);
    float sum = get_random_value();
    state.ResumeTiming();
    for (int64_t step = 0; step < state.range(1); step++) {
#pragma omp parallel reduction(+ : sum)
      for (int64_t i = 0; i < steps; i++) {
        sum += i * sum;
      }
      sum = do_something(sum);
    }
  }
}

BENCHMARK(BM_TBB_OMP) SETTING;
BENCHMARK(BM_OMP) SETTING;
BENCHMARK(BM_TBB) SETTING;
BENCHMARK_MAIN();
