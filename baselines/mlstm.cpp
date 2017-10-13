#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <locale>
#include <iostream>
#include <fstream>
#include <chrono>
#include <system_error>

#include <stdio.h>
#include <sched.h>
#include <errno.h>

#define CUDA_CHECK(result) cudaCheck(result, __FILE__, __LINE__);

static void cudaCheck(cudaError_t result, const char * file, int line) {
  if(result != cudaSuccess) {
    std::stringstream ss;
    ss << file << ":" << line << ": " << cudaGetErrorString(result);
    throw std::runtime_error(ss.str());
  }
}

inline uint64_t getTime() {
  using namespace std::chrono;
  using clock = std::conditional<high_resolution_clock::is_steady, high_resolution_clock, steady_clock>::type;
  return duration_cast<nanoseconds>(clock::now().time_since_epoch()).count();
}

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

/**
 * cpu_pin - pin down the local thread to a core
 * @cpu: the target core
 */
void cpu_pin(unsigned int cpu)
{
  int ret;
  cpu_set_t mask;

  CPU_ZERO(&mask);
  CPU_SET(cpu, &mask);

  ret = sched_setaffinity(0, sizeof(mask), &mask);
  if (ret) throw std::system_error(errno, std::system_category());
}

// Credit: https://stackoverflow.com/questions/216823/whats-the-best-way-to-trim-stdstring
static inline void rtrim(std::string &s) {
  s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
      return !std::isspace(ch);
  }).base(), s.end());
}

/**
 * Check that the power governor for a core is "performance",
 * logging a warning if it is not.
 */
void check_cpu_governor(unsigned int cpu)
{
  std::ostringstream fpss;
  fpss << "/sys/devices/system/cpu/cpu" << cpu << "/cpufreq/scaling_governor";
  std::ifstream f(fpss.str());
  if (!f.is_open()) {
    std::cerr << "WARNING: Could not find CPU " << cpu << " governor information in filesystem (are you running on Linux?)\n";
    std::cerr << "The file '" << fpss.str() << "' did not exist.\n";
  }
  std::ostringstream r;
  r << f.rdbuf();
  std::string gov(r.str());
  rtrim(gov);
  if (gov != "performance") {
    std::cerr << "WARNING: CPU " << cpu << " governor is " << gov << ", which could lead to variance in performance.\n";
    std::cerr << "Run 'echo performance > " << fpss.str() << "' as root to turn off power scaling.\n";
  }
}

// Modeled off of Soumith's benchmark at
// https://github.com/soumith/convnet-benchmarks/blob/d6177f97e61da0d98a528f355086eb2fc05fe7b8/nervana/convnet-benchmarks.py
int main() {
  constexpr unsigned int cpu = 0;

  cpu_pin(cpu);
  check_cpu_governor(cpu);

  constexpr int batch_size = 3;
  constexpr int input_size = 100;
  constexpr int hidden_size = 400;
  constexpr int embed_size = 400;
  constexpr int layers = 700;
  constexpr int loops = 10;
  constexpr int warmup = 50;

  auto input = at::CUDA(at::kFloat).randn({batch_size, input_size});
  auto hx    = at::CUDA(at::kFloat).randn({batch_size, hidden_size});
  auto cx    = at::CUDA(at::kFloat).randn({batch_size, hidden_size});
  auto w_xm  = at::CUDA(at::kFloat).randn({embed_size, input_size});
  auto w_hm  = at::CUDA(at::kFloat).randn({embed_size, hidden_size});
  auto w_ih  = at::CUDA(at::kFloat).randn({4 * hidden_size, input_size});
  auto w_mh  = at::CUDA(at::kFloat).randn({4 * hidden_size, embed_size});

  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));

  for (int i = 0; i < warmup + loops; i++) {
    CUDA_CHECK(cudaEventRecord(start, 0));
    for (int j = 0; j < layers; j++) {
      std::tie(hx, cx) = mlstm(input, hx, cx, w_xm, w_hm, w_ih, w_mh);
    }
    CUDA_CHECK(cudaEventRecord(end, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    float msecs;
    cudaEventElapsedTime(&msecs, start, end);
    printf("mlstm(%2d): %8.3f msecs\n", i, msecs);
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(end));

  return 0;
}
