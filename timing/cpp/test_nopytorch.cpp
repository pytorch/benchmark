#include "benchmark/benchmark.h"
#include <iostream>

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
    volatile float x = 1;
    for (int64_t i = 0; i < state.range(0); i++) {
      x = std::sin(x);
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
