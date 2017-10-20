#include "benchmark_common.h"

#include <ATen/ATen.h>

at::Tensor t_use(at::Tensor x) {
  return x;
}
at::Tensor t_def(at::Tensor x) {
  return x.t();
}

std::pair<at::Tensor, at::Tensor>
lstm(at::Tensor input,
      at::Tensor hx,
      at::Tensor cx,
      at::Tensor w_ih,
      at::Tensor w_hh) {
  auto gates = input.mm(t_use(w_ih)) + hx.mm(t_use(w_hh));

  auto chunked_gates = gates.chunk(4, 1);
  auto ingate     = chunked_gates[0];
  auto forgetgate = chunked_gates[1];
  auto cellgate = chunked_gates[2];
  auto outgate    = chunked_gates[3];

  ingate = ingate.sigmoid();
  outgate = outgate.sigmoid();
  cellgate = cellgate.tanh();
  forgetgate = forgetgate.sigmoid();

  auto cy = (forgetgate * cx) + (ingate * cellgate);
  auto hy = outgate * cy.tanh();

  return {hy, cy};
}

int main() {

  constexpr unsigned int cpu = 0, gpu = 0;

  cpu_pin(cpu);
  check_cpu_governor(cpu);
  check_gpu_applications_clock(gpu);

  constexpr int batch_size = 1;
  constexpr int input_size = 256;
  constexpr int hidden_size = 512;

  constexpr int fast = 0;

  constexpr int seq_len = fast ? 3 : 512;
  constexpr int warmup = fast ? 2 : 10;
  constexpr int loops  = fast ? 3 : 30;

  auto input = at::CUDA(at::kFloat).randn({seq_len, batch_size, input_size});
  auto hx    = at::CUDA(at::kFloat).randn({batch_size, hidden_size});
  auto cx    = at::CUDA(at::kFloat).randn({batch_size, hidden_size});
  auto w_ih  = t_def(at::CUDA(at::kFloat).randn({4 * hidden_size, input_size}));
  auto w_hh  = t_def(at::CUDA(at::kFloat).randn({4 * hidden_size, hidden_size}));


  // Possible experiment:
  // Create a stream that is default nonblocking
  // (don't use the default stream because shenanigans)

  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));

  for (int i = 0; i < warmup + loops; i++) {
    CUDA_CHECK(cudaEventRecord(start, 0));
    auto start_cpu = getTime();
    for (int j = 0; j < seq_len; j++) {
      std::tie(hx, cx) = lstm(input[j], hx, cx, w_ih, w_hh);
    }
    /*
    for (int j = 0; j < 300000; j++) {
      __asm__("");
    }
    */
    auto end_cpu = getTime();
    CUDA_CHECK(cudaEventRecord(end, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    float msecs;
    cudaEventElapsedTime(&msecs, start, end);
    printf("lstm(%2d): %8.3f msecs (%8.3f msecs cpu)\n", i, msecs, (end_cpu-start_cpu)/1000000.0);
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(end));

  return 0;
}
