#include <benchmark/benchmark.h>
#include <torch/torch.h>

static void BM_AtenEmpty_0(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  for (int i = 0; i < 100; i++) {
    auto tmp = at::empty({0}, options);
  }

  for (auto _ : state) {
    auto tensor = at::empty({0}, options);
  }
}
BENCHMARK(BM_AtenEmpty_0);

static void BM_VariableEmpty_0(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = torch::empty({0}, options);

  for (auto _ : state) {
    auto tensor = torch::empty({0}, options);
  }
}
BENCHMARK(BM_VariableEmpty_0);

static void BM_TensorNoopResize(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);
  std::vector<long int> sizes({64, 2048});

  // initialize some cuda...
  auto tmp = at::empty({0}, options);
  tmp.resize_(sizes);

  for (auto _ : state) {
    tmp.resize_(sizes);
  }
}
BENCHMARK(BM_TensorNoopResize);

static void BM_AtenEmpty_0_then_Resize_64_2048(
    benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);
  std::vector<long int> sizes({64, 2048});

  // initialize some cuda...
  auto tmp = at::empty({0}, options);
  tmp.resize_(sizes);

  for (auto _ : state) {
    auto tensor = at::empty({0}, options);
    tensor.resize_(sizes);
  }
}
BENCHMARK(BM_AtenEmpty_0_then_Resize_64_2048);

static void BM_AtenEmpty_64_2048(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);
  std::vector<long int> sizes({64, 2048});

  // initialize some cuda...
  auto tmp = at::empty({0}, options);
  tmp.resize_(sizes);

  for (auto _ : state) {
    auto tensor = at::empty(sizes, options);
  }
}
BENCHMARK(BM_AtenEmpty_64_2048);

static void BM_VariableEmpty_64_2048(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);
  std::vector<long int> sizes({64, 2048});
  std::vector<long int> zero({0});

  // initialize some cuda...
  auto tmp = torch::empty(zero, options);
  tmp.resize_(sizes);

  for (auto _ : state) {
    benchmark::DoNotOptimize(torch::empty(sizes, options));
  }
}
BENCHMARK(BM_VariableEmpty_64_2048);

static void BM_VariableEmpty_0_then_Resize_64_2048(
    benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);
  std::vector<long int> sizes({64, 2048});
  std::vector<long int> zero({0});

  // initialize some cuda...
  auto tmp = torch::empty(zero, options);
  tmp.resize_(sizes);

  for (auto _ : state) {
    auto tensor = torch::empty(zero, options);
    tensor.resize_(sizes);
  }
}
BENCHMARK(BM_VariableEmpty_0_then_Resize_64_2048);

BENCHMARK_MAIN();
