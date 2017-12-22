#include "benchmark_common.h"

#include <ATen/ATen.h>

int main() {

  constexpr unsigned int cpu = 0, gpu = 0;

  cpu_pin(cpu);
  check_cpu_governor(cpu);
  check_gpu_applications_clock(gpu);

  constexpr int batch_size = 1;
  //constexpr std::vector<int64_t> input_size{10, 3, 244, 244};
  const std::vector<int64_t> input_size{1, 3, 50, 50};

  constexpr int sample_size = 3;
  constexpr int warmup = 10;
  constexpr int benchmark = 30;

  auto input = at::CUDA(at::kFloat).randn(input_size);
  auto weight = at::CUDA(at::kFloat).randn({3, 3, 3, 3});

  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));

  for (int i = 0; i < warmup + benchmark; i++) {
    CUDA_CHECK(cudaEventRecord(start, 0));
    auto start_cpu_ns = getTime();
    for (int j = 0; j < sample_size; j++) {
      at::cudnn_convolution_forward(input, weight, {0, 0}, {1, 1}, {1, 1}, 1, false, false);
    }
    auto end_cpu_ns = getTime();
    CUDA_CHECK(cudaEventRecord(end, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    float gpu_msecs;
    cudaEventElapsedTime(&gpu_msecs, start, end);
    print_result_usecs("cudnn_conv", i, gpu_msecs * 1000, (end_cpu_ns-start_cpu_ns)/1000.0, sample_size);
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(end));

  return 0;
}
