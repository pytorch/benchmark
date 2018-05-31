#include "benchmark_common.h"

#include <ATen/ATen.h>

std::pair<at::Tensor, at::Tensor>
mlstm(at::Tensor input,
      at::Tensor hx,
      at::Tensor cx,
      at::Tensor w_xm,
      at::Tensor w_hm,
      at::Tensor w_ih,
      at::Tensor w_mh) {
  auto m = input.mm(w_xm.t()) * hx.mm(w_hm.t());
  auto gates = input.mm(w_ih.t()) + m.mm(w_mh.t());

  auto chunked_gates = gates.chunk(4, 1);
  auto ingate     = chunked_gates[0];
  auto forgetgate = chunked_gates[1];
  auto hiddengate = chunked_gates[2];
  auto outgate    = chunked_gates[3];

  ingate = ingate.sigmoid();
  outgate = outgate.sigmoid();
  forgetgate = forgetgate.sigmoid();

  auto cy = (forgetgate * cx) + (ingate * hiddengate);
  auto hy = (cy * outgate).tanh();

  return {hy, cy};
}

// Modeled off of Soumith's benchmark at
// https://github.com/soumith/convnet-benchmarks/blob/d6177f97e61da0d98a528f355086eb2fc05fe7b8/nervana/convnet-benchmarks.py
int main() {

  constexpr unsigned int cpu = 0, gpu = 0;

  cpu_pin(cpu);
  check_cpu_governor(cpu);
  check_gpu_applications_clock(gpu);

  // TODO: Check power state?  Applications clock might be enough

  // Parameters taken from Hutter Prize dataset
  // "The dataset is modelled using a UTF-8 encoding, and contains 205 unique
  // bytes."
  // "The weight normalized LSTM used 1900 hidden units, and a linear embedding
  // layer of 400, giving it 22 million parameters."

  constexpr int batch_size = 1;     // arbitrary
  constexpr int input_size = 205;   // number of bytes
  constexpr int hidden_size = 1900; // hidden units
  // "We set the dimensionality of m_t and h_t equal for all our experiments." (pg. 4)
  constexpr int embed_size = hidden_size;

  constexpr int fast = 0;

  constexpr int seq_len = fast ? 3 : 20; // truncated backpropagation length
  constexpr int loops  = fast ? 3 : 30;
  constexpr int warmup = fast ? 2 : 10;

  auto input = at::CUDA(at::kFloat).randn({batch_size, input_size});
  auto hx    = at::CUDA(at::kFloat).randn({batch_size, hidden_size});
  auto cx    = at::CUDA(at::kFloat).randn({batch_size, hidden_size});
  auto w_xm  = at::CUDA(at::kFloat).randn({embed_size, input_size});
  auto w_hm  = at::CUDA(at::kFloat).randn({embed_size, hidden_size});
  auto w_ih  = at::CUDA(at::kFloat).randn({4 * hidden_size, input_size});
  auto w_mh  = at::CUDA(at::kFloat).randn({4 * hidden_size, embed_size});


  // Possible experiment:
  // Create a stream that is default nonblocking
  // (don't use the default stream because shenanigans)
  //
  // Experiment: run this on devgpu with numa pinning (even though there's noise)
  //
  // Experiment: look in nvvp

  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));

  for (int i = 0; i < warmup + loops; i++) {
    CUDA_CHECK(cudaEventRecord(start, 0));
    auto start_cpu_ns = getTime();
    for (int j = 0; j < seq_len; j++) {
      std::tie(hx, cx) = mlstm(input, hx, cx, w_xm, w_hm, w_ih, w_mh);
    }
    auto end_cpu_ns = getTime();
    CUDA_CHECK(cudaEventRecord(end, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    float gpu_msecs;
    cudaEventElapsedTime(&gpu_msecs, start, end);
    print_result_usecs("mlstm", i, gpu_msecs * 1000, (end_cpu_ns-start_cpu_ns)/1000.0, seq_len);
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(end));

  return 0;
}
