#include "ATen/ATen.h"
#include "ATen/Parallel.h"
#include "benchmark/benchmark.h"
#include <iostream>

using namespace at;

#define BASIC_BENCHMARK_TEST(x) BENCHMARK(x)->Arg(8)->Arg(512)->Arg(8192)

void BM_empty(benchmark::State &state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(state.iterations());
  }
}
BENCHMARK(BM_empty);
BENCHMARK(BM_empty)->ThreadPerCpu();

void BM_spin_empty(benchmark::State &state) {
  for (auto _ : state) {
    auto t = randn(CPU(kFloat), {(int64_t)(1e6)});
    for (int64_t i = 0; i < state.range(0); i++) {
      t.sin_();
    }
  }
}
BASIC_BENCHMARK_TEST(BM_spin_empty);
BASIC_BENCHMARK_TEST(BM_spin_empty)->ThreadPerCpu();

// Ensure that StateIterator provides all the necessary typedefs required to
// instantiate std::iterator_traits.
static_assert(
    std::is_same<typename std::iterator_traits<
                     benchmark::State::StateIterator>::value_type,
                 typename benchmark::State::StateIterator::value_type>::value,
    "");

BENCHMARK_MAIN();
