#include "benchmark_common.h"

// #include <ATen/ATen.h>

#include <cudnn.h>

int main() {

  constexpr unsigned int cpu = 0, gpu = 0;

  cpu_pin(cpu);
  check_cpu_governor(cpu);
  check_gpu_applications_clock(gpu);

  constexpr int sample_size = 1000000;
  constexpr int warmup = 2;
  constexpr int benchmark = 4;

  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));

  for (int i = 0; i < warmup + benchmark; i++) {
    CUDA_CHECK(cudaEventRecord(start, 0));
    auto start_cpu_ns = getTime();
    for (int j = 0; j < sample_size; j++) {
      cudnnTensorDescriptor_t desc;
      CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
      CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc));
      //cudnnDropoutDescriptor_t dropout_desc;
      //CUDNN_CHECK(cudnnCreateDropoutDescriptor(&dropout_desc));
      //CUDNN_CHECK(cudnnDestroyDropoutDescriptor(dropout_desc));
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
