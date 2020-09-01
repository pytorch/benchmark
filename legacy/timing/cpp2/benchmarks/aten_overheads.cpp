#include <iostream>
#include <benchmark/benchmark.h>
#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ExpandUtils.h>
#include <c10/cuda/CUDAFunctions.h>

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

static void BM_InferType(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({64, 2048}, options);

  for (auto _ : state) {
    at::detail::infer_type(tmp);
  }
}
BENCHMARK(BM_InferType);

static void BM_SmallVectorAssign1(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({64, 2048}, options);
  std::vector<int64_t> foo({1, 2, 3, 4, 5});
  at::IntList fooref = foo;
  at::SmallVector<int64_t,5> smallvec;
  smallvec.resize(5);

  for (auto _ : state) {
    smallvec = fooref;
  }
}
BENCHMARK(BM_SmallVectorAssign1);

static void BM_SmallVectorAssign2(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({64, 2048}, options);
  std::vector<int64_t> foo({1, 2, 3, 4, 5});
  at::IntList fooref = foo;
  at::SmallVector<int64_t,5> smallvec;
  smallvec.resize(5);

  for (auto _ : state)
    smallvec = foo;
}
BENCHMARK(BM_SmallVectorAssign2);

static void BM_SmallVectorAssign3(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({64, 2048}, options);
  std::vector<int64_t> foo({1, 2, 3, 4, 5});
  at::IntList fooref = foo;
  at::SmallVector<int64_t,5> smallvec;
  smallvec.resize(5);

  for (auto _ : state) {
    smallvec.resize(5);
    for (int64_t i = 0; i < foo.size(); ++i) {
      smallvec[i] = foo[i];
    }
  }
}
BENCHMARK(BM_SmallVectorAssign3);

static void BM_SmallVectorAssign4(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({64, 2048}, options);
  std::vector<int64_t> foo({1, 2, 3, 4, 5});
  at::IntList fooref = foo;
  at::SmallVector<int64_t,5> smallvec;
  smallvec.resize(5);

  for (auto _ : state) {
    smallvec.assign(foo.begin(), foo.end());
  }
}
BENCHMARK(BM_SmallVectorAssign4);

static void BM_SmallVectorAssign5(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({64, 2048}, options);
  std::vector<int64_t> foo({1, 2, 3, 4, 5});
  at::IntList fooref = foo;
  at::SmallVector<int64_t,5> smallvec;
  smallvec.resize(5);

  for (auto _ : state) {
    smallvec = fooref.vec();
  }
}
BENCHMARK(BM_SmallVectorAssign5);

static void BM_VectorAllocation(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({64, 2048}, options);

  for (auto _ : state) {
    benchmark::DoNotOptimize(std::vector<int64_t>(5));
  }
}
BENCHMARK(BM_VectorAllocation);

static void BM_SmallVectorAllocation(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({64, 2048}, options);

  for (auto _ : state)
    benchmark::DoNotOptimize(at::SmallVector<int64_t,5>());
}
BENCHMARK(BM_SmallVectorAllocation);

static void BM_IntArrayAllocation(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({64, 2048}, options);

  for (auto _ : state) {
    // yes, this leaks memory
    benchmark::DoNotOptimize(new int64_t[3]);
  }
}
BENCHMARK(BM_IntArrayAllocation);


static void BM_THCCachingAllocatorAllocate(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  int size = 64 * 2048;
  auto tmp = at::empty({size}, options);
  auto* impl = tmp.unsafeGetTensorImpl();

  // allocate memory once so that caching allocator has it.
  {
    at::DataPtr data = impl->storage().allocator()->allocate(size * 4);
  }

  for (auto _ : state) {
    at::DataPtr data = impl->storage().allocator()->allocate(size * 4);
  }
}
BENCHMARK(BM_THCCachingAllocatorAllocate);

static void BM_CudaAPIGetDevice(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);
  int32_t device;

  for (auto _ : state) {
    benchmark::DoNotOptimize(cudaGetDevice(&device));
  }
}
BENCHMARK(BM_CudaAPIGetDevice);

static void BM_CudaAPISetDevice(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);
  int32_t device;
  cudaGetDevice(&device);

  for (auto _ : state) {
    benchmark::DoNotOptimize(cudaSetDevice(device));
  }
}
BENCHMARK(BM_CudaAPISetDevice);

static void BM_DynamicCUDAInterfaceGetDevice(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  for (auto _ : state) {
    c10::cuda::current_device();
  }
}
BENCHMARK(BM_DynamicCUDAInterfaceGetDevice);

static void BM_DynamicCUDAInterfaceSetDevice(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);
  int32_t device;
  cudaGetDevice(&device);

  for (auto _ : state) {
    c10::cuda::set_device(device);
  }
}
BENCHMARK(BM_DynamicCUDAInterfaceSetDevice);

static void BM_Mutex(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);
  int32_t device;
  cudaGetDevice(&device);

  std::mutex mutex;

  for (auto _ : state) {
  }
}
BENCHMARK(BM_Mutex);

static void BM_GetCurrentCUDAStream(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);
  int32_t device;
  cudaGetDevice(&device);

  for (auto _ : state) {
    benchmark::DoNotOptimize(at::cuda::getCurrentCUDAStream(device));
  }
}
BENCHMARK(BM_GetCurrentCUDAStream);

static void BM_StorageImplGetDevice(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);
  auto* storage_impl = tmp.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl();

  for (auto _ : state) {
    benchmark::DoNotOptimize(storage_impl->device().index());
  }
}
BENCHMARK(BM_StorageImplGetDevice);

static void BM_TensorImplGetDevice(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);
  auto* tensor_impl = tmp.unsafeGetTensorImpl();

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        tensor_impl->storage().unsafeGetStorageImpl()->device().index());
  }
}
BENCHMARK(BM_TensorImplGetDevice);

static void BM_VariableGetDevice(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = torch::empty({0}, options);

  for (auto _ : state) {
    benchmark::DoNotOptimize(tmp.get_device());
  }

}
BENCHMARK(BM_VariableGetDevice);

//static void BM_DeviceGuardCtor(benchmark::State& state) {
//  auto options = at::TensorOptions(at::kCUDA);
//
//  // initialize some cuda...
//  auto tmp = at::empty({0}, options);
//  void* mem = malloc(sizeof(at::DeviceGuard));
//
//  for (auto _ : state) {
//    benchmark::DoNotOptimize(new (mem) at::DeviceGuard(tmp));
//  }
//
//  free(mem);
//}
//BENCHMARK(BM_DeviceGuardCtor);
//
static void BM_DeviceGuard(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  for (auto _ : state) {
    {
      const at::OptionalDeviceGuard guard(device_of(tmp));
    }
  }
}
BENCHMARK(BM_DeviceGuard);

static void BM_DeviceGuardFromDevice(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);
  auto device = options.device();

  for (auto _ : state) {
    {
      const at::DeviceGuard guard(device);
    }
  }
}
BENCHMARK(BM_DeviceGuardFromDevice);


static void BM_CheckedTensorUnwrap(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        at::checked_tensor_unwrap(tmp,"self",1, false, at::Backend::CUDA, at::ScalarType::Float));
  }
}
BENCHMARK(BM_CheckedTensorUnwrap);

BENCHMARK_MAIN();

