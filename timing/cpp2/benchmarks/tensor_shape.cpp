#include <benchmark/benchmark.h>
#include <torch/torch.h>
#include <ATen/ExpandUtils.h>

static void BM_TensorOptions(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({64, 2048}, options);

  for (auto _ : state)
    tmp.options();
}
BENCHMARK(BM_TensorOptions);

static void BM_TensorAsStridedInplace(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({64, 2048}, options);
  std::vector<long int> sizes1({2048*64});
  std::vector<long int> strides1({1});
  std::vector<long int> sizes2({64, 2048});
  std::vector<long int> strides2({2048, 1});

  bool flag = true;

  for (auto _ : state) {
    // prevent no-op optimizations
    if (flag) {
      tmp.as_strided_(sizes1, strides1);
    } else {
      tmp.as_strided_(sizes2, strides2);
    }
    flag = !flag;
  }
}
BENCHMARK(BM_TensorAsStridedInplace);

static void BM_TensorAsStridedOffsetInplace(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({64, 2048}, options);
  std::vector<long int> sizes1({2048*64});
  std::vector<long int> strides1({1});
  std::vector<long int> sizes2({64, 2048});
  std::vector<long int> strides2({2048, 1});

  bool flag = true;

  for (auto _ : state) {
    // prevent no-op optimizations
    if (flag) {
      tmp.as_strided_(sizes1, strides1, 0);
    } else {
      tmp.as_strided_(sizes2, strides2, 0);
    }
    flag = !flag;
  }
}
BENCHMARK(BM_TensorAsStridedOffsetInplace);

static void BM_AtenEmpty_0(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  for (auto _ : state) {
    auto tensor = at::empty({0}, options);
  }
}
BENCHMARK(BM_AtenEmpty_0);

static void BM_AtenEmptyOptions_0(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  for (auto _ : state) {
    auto tensor = at::empty({0}, tmp.options());
  }
}
BENCHMARK(BM_AtenEmptyOptions_0);


static void BM_TensorImplSetStorage(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({62, 2048}, options);
  auto s1 = at::empty({64, 2048}, options);
  auto s2 = at::empty({64, 2048}, options);
  auto storage1 = s1.storage();
  auto storage2 = s2.storage();
  auto* impl = tmp.unsafeGetTensorImpl();

  bool flag = true;

  for (auto _ : state) {
    // prevent no-op optimizations
    if (flag) {
      impl->set_storage(storage1);
    } else {
      impl->set_storage(storage2);
    }
    flag = !flag;
  }
}
BENCHMARK(BM_TensorImplSetStorage);

static void BM_TensorImplSetSizesStrides(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({64, 2048}, options);
  std::vector<long int> sizes1({2048*64});
  std::vector<long int> strides1({1});
  std::vector<long int> sizes2({64, 2048});
  std::vector<long int> strides2({2048, 1});
  auto* impl = tmp.unsafeGetTensorImpl();

  bool flag = true;

  for (auto _ : state) {
    // prevent no-op optimizations
    if (flag) {
      impl->set_sizes_and_strides(sizes1, strides1);
    } else {
      impl->set_sizes_and_strides(sizes2, strides2);
    }
    flag = !flag;
  }
}
BENCHMARK(BM_TensorImplSetSizesStrides);

static void BM_TensorNoopSetStorageOffset(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({64, 2048}, options);
  auto* impl = tmp.unsafeGetTensorImpl();

  for (auto _ : state)
    impl->set_storage_offset(0);
}
BENCHMARK(BM_TensorNoopSetStorageOffset);

static void BM_TensorAsStrided(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);
  auto tensor = at::rand({2400}, options);
  std::vector<long int> strides({1, 300});
  std::vector<long int> sizes({300, 8});

  for (auto _ : state)
    benchmark::DoNotOptimize(tensor.as_strided(strides, sizes));
}
BENCHMARK(BM_TensorAsStrided);

static void BM_TensorAsStridedOffset(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);
  auto tensor = at::rand({2400}, options);
  std::vector<long int> strides({1, 300});
  std::vector<long int> sizes({300, 8});

  for (auto _ : state)
    benchmark::DoNotOptimize(tensor.as_strided(strides, sizes, 0));
}
BENCHMARK(BM_TensorAsStridedOffset);


static void BM_TensorNoopAsStrided(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({64, 2048}, options);
  std::vector<long int> strides({2048, 1});
  std::vector<long int> sizes({64, 2048});

  for (auto _ : state)
    tmp.as_strided_(sizes, strides);
}
BENCHMARK(BM_TensorNoopAsStrided);

static void BM_TensorExpand(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({2048}, options);

  for (auto _ : state) {
    benchmark::DoNotOptimize(tmp.expand({64, 2048}));
  }
}
BENCHMARK(BM_TensorExpand);

static void BM_TensorInferExpandGeometry(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({2048}, options);
  auto sizes = at::IntList({64, 2048});

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        inferExpandGeometry(tmp.sizes(), tmp.strides(), sizes));
  }
}
BENCHMARK(BM_TensorInferExpandGeometry);

BENCHMARK_MAIN();
