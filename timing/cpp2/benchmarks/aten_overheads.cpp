#include <iostream>
#include <benchmark/benchmark.h>
#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>

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

static void BM_TensorTypeIsCuda(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  for (auto _ : state) {
    benchmark::DoNotOptimize(tmp.type().is_cuda());
  }
}
BENCHMARK(BM_TensorTypeIsCuda);

static void BM_TensorNumel(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  for (auto _ : state) {
    benchmark::DoNotOptimize(tmp.numel());
  }
}
BENCHMARK(BM_TensorNumel);

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
  int32_t device;

  for (auto _ : state) {
    at::detail::DynamicCUDAInterface::get_device(&device);
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
    at::detail::DynamicCUDAInterface::set_device(device);
  }
}
BENCHMARK(BM_DynamicCUDAInterfaceSetDevice);

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

static void BM_TensorGetDeviceDirect(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        tmp.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->device().index());
  }
}
BENCHMARK(BM_TensorGetDeviceDirect);


//static void BM_THGetDevice(benchmark::State& state) {
//  auto options = at::TensorOptions(at::kCUDA);
//
//  // initialize some cuda...
//  auto tmp = at::empty({0}, options);
//
//  for (auto _ : state) {
//    benchmark::DoNotOptimize(at::_th_get_device(tmp));
//  }
//
//}
//BENCHMARK(BM_THGetDevice);

static void BM_TensorGetDevice(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  for (auto _ : state) {
    benchmark::DoNotOptimize(tmp.get_device());
  }

}
BENCHMARK(BM_TensorGetDevice);

static void BM_DeviceGuardCtor(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);
  void* mem = malloc(sizeof(at::DeviceGuard));

  for (auto _ : state) {
    benchmark::DoNotOptimize(new (mem) at::DeviceGuard(tmp));
  }

  free(mem);
}
BENCHMARK(BM_DeviceGuardCtor);

static void BM_DeviceGuard(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  for (auto _ : state) {
    {
      const at::DeviceGuard guard(tmp);
    }
  }
}
BENCHMARK(BM_DeviceGuard);

static void BM_EmptyTensorNoopResize(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);
  std::vector<long int> sizes({0});

  // initialize some cuda...
  auto tmp = at::empty({0}, options);
  tmp.resize_(sizes);

  for (auto _ : state) {
    tmp.resize_(sizes);
  }
}
BENCHMARK(BM_EmptyTensorNoopResize);

//static void BM_NoopEmptyResizeNoDispatch(benchmark::State& state) {
//  auto options = at::TensorOptions(at::kCUDA);
//  std::vector<long int> sizes({0});
//
//  // initialize some cuda...
//  auto tmp = at::empty({0}, options);
//  tmp.resize_(sizes);
//
//  for (auto _ : state) {
//    at::native::resize__cuda(tmp, sizes);
//  }
//}
//BENCHMARK(BM_NoopEmptyResizeNoDispatch);

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

static void BM_TensorAsStrided(benchmark::State& state) {
  auto tensor = at::rand({2400});
  std::vector<long int> strides({1, 300});
  std::vector<long int> sizes({300, 8});

  for (auto _ : state)
    benchmark::DoNotOptimize(tensor.as_strided(strides, sizes));
}
BENCHMARK(BM_TensorAsStrided);

static void BM_AtenEmptyCuda(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  for (auto _ : state) {
    auto tensor = at::native::empty_cuda({0}, options);
  }
}
BENCHMARK(BM_AtenEmptyCuda);

static void BM_AtenEmpty(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = at::empty({0}, options);

  for (auto _ : state) {
    auto tensor = at::empty({0}, options);
  }
}
BENCHMARK(BM_AtenEmpty);

static void BM_VariableEmpty(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = torch::empty({0}, options);

  for (auto _ : state) {
    auto tensor = torch::empty({0}, options);
  }
}
BENCHMARK(BM_VariableEmpty);

static void BM_AtenEmptyResize(benchmark::State& state) {
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
BENCHMARK(BM_AtenEmptyResize);

static void BM_AtenEmptyNoResize(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);
  std::vector<long int> sizes({64, 2048});

  // initialize some cuda...
  auto tmp = at::empty({0}, options);
  tmp.resize_(sizes);

  for (auto _ : state) {
    auto tensor = at::empty(sizes, options);
  }
}
BENCHMARK(BM_AtenEmptyNoResize);


static void BM_VariableEmptyResize(benchmark::State& state) {
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
BENCHMARK(BM_VariableEmptyResize);

static void BM_VariableEmptyNoResize(benchmark::State& state) {
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
BENCHMARK(BM_VariableEmptyNoResize);


static void BM_MakeStorage(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = torch::empty({0}, options);

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        c10::make_intrusive<at::StorageImpl>(
            at::scalarTypeToTypeMeta(options.dtype()),
            0,
            at::cuda::getCUDADeviceAllocator(),
            true));
  }
}
BENCHMARK(BM_MakeStorage);

static void BM_StorageCtor(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = torch::empty({0}, options);

  void* mem = malloc(sizeof(at::StorageImpl));

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        new (mem) at::StorageImpl(
            at::scalarTypeToTypeMeta(options.dtype()),
            0,
            at::cuda::getCUDADeviceAllocator(),
            true));
  }

  free(mem);
}
BENCHMARK(BM_StorageCtor);

static void BM_MallocOverhead(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(malloc(1));
  }
}
BENCHMARK(BM_MallocOverhead);

static void BM_StorageMalloc(benchmark::State& state) {
  for (auto _ : state) {
    // NB: leaks memory
    benchmark::DoNotOptimize(malloc(sizeof(at::StorageImpl)));
  }
}
BENCHMARK(BM_StorageMalloc);

static void BM_ScalarTypeToTypeMeta(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = torch::empty({0}, options);

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        at::scalarTypeToTypeMeta(options.dtype()));
  }
}
BENCHMARK(BM_ScalarTypeToTypeMeta);

static void BM_MakeTensorFromStorage(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = torch::empty({0}, options);

  auto storage = c10::make_intrusive<at::StorageImpl>(
            at::scalarTypeToTypeMeta(options.dtype()),
            0,
            at::cuda::getCUDADeviceAllocator(),
            true);

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        at::detail::make_tensor<at::TensorImpl>(storage, at::CUDATensorId(), false));
  }
}
BENCHMARK(BM_MakeTensorFromStorage);

static void BM_MakeVariableFromTensor(benchmark::State& state) {
  auto options = at::TensorOptions(at::kCUDA);

  // initialize some cuda...
  auto tmp = torch::empty({0}, options);

  auto storage_impl = c10::make_intrusive<at::StorageImpl>(
            at::scalarTypeToTypeMeta(options.dtype()),
            0,
            at::cuda::getCUDADeviceAllocator(),
            true);
  auto tensor = at::detail::make_tensor<at::TensorImpl>(
      storage_impl, at::CUDATensorId(), false);

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        torch::autograd::make_variable(tensor, false));
  }
}
BENCHMARK(BM_MakeVariableFromTensor);



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

