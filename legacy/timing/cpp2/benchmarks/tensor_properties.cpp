#include <benchmark/benchmark.h>
#include <torch/torch.h>

static void BM_TensorSizes(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({2048}, options);

  for (auto _ : state) {
    benchmark::DoNotOptimize(tmp.sizes());
  }
}
BENCHMARK(BM_TensorSizes);

static void BM_TensorStrides(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({2048}, options);

  for (auto _ : state) {
    benchmark::DoNotOptimize(tmp.strides());
  }
}
BENCHMARK(BM_TensorStrides);

static void BM_TensorOptions(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({64, 2048}, options);

  for (auto _ : state)
    tmp.options();
}
BENCHMARK(BM_TensorOptions);

static void BM_TensorOptionEmpty(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({64, 2048}, options);

  for (auto _ : state)
    benchmark::DoNotOptimize(at::TensorOptions());
}
BENCHMARK(BM_TensorOptionEmpty);

static void BM_TensorDtype(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({64, 2048}, options);

  for (auto _ : state)
    benchmark::DoNotOptimize(tmp.dtype());
}
BENCHMARK(BM_TensorDtype);

static void BM_TensorDevice(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({64, 2048}, options);

  for (auto _ : state)
    benchmark::DoNotOptimize(tmp.device());
}
BENCHMARK(BM_TensorDevice);

static void BM_TensorLayout(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({64, 2048}, options);

  for (auto _ : state)
    benchmark::DoNotOptimize(tmp.layout());
}
BENCHMARK(BM_TensorLayout);

static void BM_TensorStorageOffset(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({64, 2048}, options);

  for (auto _ : state)
    tmp.storage_offset();
}
BENCHMARK(BM_TensorStorageOffset);

static void BM_TensorIsCuda(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  for (auto _ : state) {
    benchmark::DoNotOptimize(tmp.is_cuda());
  }
}
BENCHMARK(BM_TensorIsCuda);

static void BM_TensorDim(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  for (auto _ : state) {
    benchmark::DoNotOptimize(tmp.dim());
  }
}
BENCHMARK(BM_TensorDim);

static void BM_TensorIsSparse(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  for (auto _ : state) {
    benchmark::DoNotOptimize(tmp.is_sparse());
  }
}
BENCHMARK(BM_TensorIsSparse);

static void BM_TensorIsVariable(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  for (auto _ : state) {
    benchmark::DoNotOptimize(tmp.is_variable());
  }
}
BENCHMARK(BM_TensorIsVariable);

static void BM_TensorNumel(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  for (auto _ : state) {
    benchmark::DoNotOptimize(tmp.numel());
  }
}
BENCHMARK(BM_TensorNumel);

static void BM_TensorGetDevice(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  for (auto _ : state) {
    benchmark::DoNotOptimize(tmp.get_device());
  }
}
BENCHMARK(BM_TensorGetDevice);

static void BM_TensorStorage(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({64, 2048}, options);

  for (auto _ : state)
    tmp.storage();
}
BENCHMARK(BM_TensorStorage);

static void BM_TensorTypeId(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);
  std::vector<long int> sizes({64, 2048});

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  for (auto _ : state) {
    tmp.unsafeGetTensorImpl()->type_id();
  }
}
BENCHMARK(BM_TensorTypeId);

static void BM_TensorType(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);
  std::vector<long int> sizes({64, 2048});

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  for (auto _ : state) {
    tmp.type();
  }
}
BENCHMARK(BM_TensorType);


BENCHMARK_MAIN();
