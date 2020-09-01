#include <benchmark/benchmark.h>
#include <torch/torch.h>
#include <ATen/ExpandUtils.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/csrc/autograd/functions/utils.h>

// Global variables.
constexpr size_t kBiggerThanCacheSize = 51200 * 1024;
long* kBuffer = new long[kBiggerThanCacheSize];

static inline void clearCache(benchmark::State& state) {
  state.PauseTiming();
	for (int i = 0; i < kBiggerThanCacheSize; ++i) {
	  kBuffer[i]++;
  }
  state.ResumeTiming();
}

static void BM_VariableAsStridedOffset(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);
  auto tensor = torch::rand({2400}, options);
  std::vector<long int> strides({1, 300});
  std::vector<long int> sizes({300, 8});

  for (auto _ : state)
    benchmark::DoNotOptimize(tensor.as_strided(strides, sizes, 0));
}
BENCHMARK(BM_VariableAsStridedOffset);

static void BM_VariableAsStridedAlt(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);
  auto self = torch::rand({2400}, options);
  std::vector<long int> strides({1, 300});
  std::vector<long int> sizes({300, 8});

  for (auto _ : state) {
    if (!self.defined()) {
      throw std::runtime_error("yikes");
    }
    if (!self.type().is_variable()) {
      throw std::runtime_error("yikes2");
    }
    std::shared_ptr<torch::autograd::generated::AsStridedBackward> grad_fn;
    if (torch::autograd::compute_requires_grad(self)) {
      throw std::runtime_error("yikes3");
    }
    auto& self_ = torch::autograd::as_variable_ref(self).data();
    if (torch::jit::tracer::isTracing()) {
      throw std::runtime_error("yikes4");
    }
    auto result = torch::autograd::as_view(
        self, self_.as_strided(sizes, strides, 0), true);
    torch::autograd::set_history(torch::autograd::flatten_tensor_args( result ), grad_fn);
  }
}
BENCHMARK(BM_VariableAsStridedAlt);

static void BM_FlattenTensorArgs(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);
  auto self = torch::rand({2400}, options);

  for (auto _ : state) {
    benchmark::DoNotOptimize(torch::autograd::flatten_tensor_args( self ));
  }
}
BENCHMARK(BM_FlattenTensorArgs);

static void BM_VariableAsView(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);
  auto self = torch::rand({2400}, options);
  std::vector<long int> strides({1, 300});
  std::vector<long int> sizes({300, 8});

  if (!self.defined()) {
    throw std::runtime_error("yikes");
  }
  if (!self.is_variable()) {
    throw std::runtime_error("yikes2");
  }
  std::shared_ptr<torch::autograd::generated::AsStridedBackward> grad_fn;
  if (torch::autograd::compute_requires_grad(self)) {
    throw std::runtime_error("yikes3");
  }
  auto& self_ = torch::autograd::as_variable_ref(self).data();
  if (torch::jit::tracer::isTracing()) {
    throw std::runtime_error("yikes4");
  }
  auto res = self_.as_strided(sizes, strides, 0);

  for (auto _ : state) {
    auto result = torch::autograd::as_view(self, res, true);
  }
}
BENCHMARK(BM_VariableAsView);


static void BM_VariableAsStridedAlt2(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);
  auto self = torch::rand({2400}, options);
  std::vector<long int> strides({1, 300});
  std::vector<long int> sizes({300, 8});

  for (auto _ : state) {
    if (!self.defined()) {
      throw std::runtime_error("yikes");
    }
    if (!self.is_variable()) {
      throw std::runtime_error("yikes2");
    }
    std::shared_ptr<torch::autograd::generated::AsStridedBackward> grad_fn;
    if (torch::autograd::compute_requires_grad(self)) {
      throw std::runtime_error("yikes3");
    }
    auto& self_ = torch::autograd::as_variable_ref(self).data();
    if (torch::jit::tracer::isTracing()) {
      throw std::runtime_error("yikes4");
    }
    auto res = self_.as_strided(sizes, strides, 0);
    auto result = torch::autograd::as_view(self, res, true);
    if (grad_fn) {
      torch::autograd::set_history(torch::autograd::flatten_tensor_args( result ), grad_fn);
    }
  }
}
BENCHMARK(BM_VariableAsStridedAlt2);

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

static void BM_AtenEmptyDim2(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({30, 80}, options);

  for (auto _ : state) {
    auto tensor = at::empty({30, 80}, options);
  }
}
BENCHMARK(BM_AtenEmptyDim2);

static void BM_AtenEmptyDim3(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({30, 8, 10}, options);

  for (auto _ : state) {
    auto tensor = at::empty({30, 8, 10}, options);
  }
}
BENCHMARK(BM_AtenEmptyDim3);

static void BM_AtenEmptyDim4(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({30, 80}, options);

  for (auto _ : state) {
    auto tensor = at::empty({3, 10, 8, 10}, options);
  }
}
BENCHMARK(BM_AtenEmptyDim4);

static void BM_AtenEmptyDim5(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({30, 80}, options);

  for (auto _ : state) {
    auto tensor = at::empty({3, 10, 8, 2, 5}, options);
  }
}
BENCHMARK(BM_AtenEmptyDim5);

static void BM_AtenEmptyDim6(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({30, 80}, options);

  for (auto _ : state) {
    auto tensor = at::empty({3, 2, 5, 8, 2, 5}, options);
  }
}
BENCHMARK(BM_AtenEmptyDim6);

static void BM_AtenEmptyDim7(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({30, 80}, options);

  for (auto _ : state) {
    auto tensor = at::empty({3, 2, 5, 2, 4, 2, 5}, options);
  }
}
BENCHMARK(BM_AtenEmptyDim7);

static void BM_TensorAsStridedDim2(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);
  auto tmp = at::rand({30, 80}, options);
  std::vector<long int> strides(tmp.strides().vec());
  std::vector<long int> sizes(tmp.sizes().vec());

  auto tensor = at::rand({2400}, options);

  for (auto _ : state)
    benchmark::DoNotOptimize(tensor.as_strided(sizes, strides));
}
BENCHMARK(BM_TensorAsStridedDim2);

static void BM_TensorAsStridedDim3(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);
  auto tmp = at::rand({30, 8, 10}, options);
  std::vector<long int> strides(tmp.strides().vec());
  std::vector<long int> sizes(tmp.sizes().vec());

  auto tensor = at::rand({2400}, options);

  for (auto _ : state)
    benchmark::DoNotOptimize(tensor.as_strided(sizes, strides));
}
BENCHMARK(BM_TensorAsStridedDim3);

static void BM_TensorAsStridedDim4(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);
  auto tmp = at::rand({3, 10, 8, 10}, options);
  std::vector<long int> strides(tmp.strides().vec());
  std::vector<long int> sizes(tmp.sizes().vec());

  auto tensor = at::rand({2400}, options);

  for (auto _ : state)
    benchmark::DoNotOptimize(tensor.as_strided(sizes, strides));
}
BENCHMARK(BM_TensorAsStridedDim4);

static void BM_TensorAsStridedDim5(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);
  auto tmp = at::rand({3, 10, 8, 2, 5}, options);
  std::vector<long int> strides(tmp.strides().vec());
  std::vector<long int> sizes(tmp.sizes().vec());

  auto tensor = at::rand({2400}, options);

  for (auto _ : state)
    benchmark::DoNotOptimize(tensor.as_strided(sizes, strides));
}
BENCHMARK(BM_TensorAsStridedDim5);

static void BM_TensorAsStridedDim6(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);
  auto tmp = at::rand({3, 2, 5, 8, 2, 5}, options);
  std::vector<long int> strides(tmp.strides().vec());
  std::vector<long int> sizes(tmp.sizes().vec());

  auto tensor = at::rand({2400}, options);

  for (auto _ : state)
    benchmark::DoNotOptimize(tensor.as_strided(sizes, strides));
}
BENCHMARK(BM_TensorAsStridedDim6);

static void BM_TensorAsStridedDim7(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);
  auto tmp = at::rand({3, 2, 5, 2, 4, 2, 5}, options);
  std::vector<long int> strides(tmp.strides().vec());
  std::vector<long int> sizes(tmp.sizes().vec());

  auto tensor = at::rand({2400}, options);

  for (auto _ : state)
    benchmark::DoNotOptimize(tensor.as_strided(sizes, strides));
}
BENCHMARK(BM_TensorAsStridedDim7);

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

static void BM_VariableAsStrided(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);
  auto tensor = torch::rand({2400}, options);
  std::vector<long int> strides({1, 300});
  std::vector<long int> sizes({300, 8});

  for (auto _ : state)
    benchmark::DoNotOptimize(tensor.as_strided(strides, sizes));
}
BENCHMARK(BM_VariableAsStrided);

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
