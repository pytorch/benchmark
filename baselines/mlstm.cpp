#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvml.h>

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

/*
Build with:
LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CONDA_PREFIX/lib g++ mlstm.cpp -I$CUDA_HOME/include -I$CONDA_PREFIX/include -L$CUDA_HOME/lib64 -L$CUDA_HOME/lib64/stubs -L$CONDA_PREFIX/lib -lnvidia-ml -lcudart -lATen -std=c++11 -O3 -Wall -Wextra -pedantic

Run with:
LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CONDA_PREFIX/lib ./a.out
*/

#define CUDA_CHECK(result) cudaCheck(result, __FILE__, __LINE__)
#define NVML_CHECK(result) nvmlCheck(result, __FILE__, __LINE__)

static void cudaCheck(cudaError_t result, const char * file, int line) {
  if(result != cudaSuccess) {
    std::stringstream ss;
    ss << file << ":" << line << ": " << cudaGetErrorString(result);
    throw std::runtime_error(ss.str());
  }
}

static void nvmlCheck(nvmlReturn_t result, const char * file, int line) {
  if(result != NVML_SUCCESS) {
    std::stringstream ss;
    ss << file << ":" << line << ": " << nvmlErrorString(result);
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

// The 'applications clock', also (confusingly) known as 'GPU boost', is a
// mechanism for changing the base processor clock speed.  Although this feature
// is normally advertised as a way to increase performance, it is also useful
// way stabilizing benchmark numbers.  The reason: when the GPU's base clock
// speed (the minimum clock speed it's willing to run at) is lower than its max
// clock speed, if there is power/thermal headroom, they will "boost" their
// clock to take advantage of this extra capacity; this is good for normal users
// but bad for benchmarking, since the variable clock speed introduces
// variability to timings.
//
// Source:
// https://devblogs.nvidia.com/parallelforall/increase-performance-gpu-boost-k80-autoboost/
//
// However, there is one hazard to setting things up this way: if the driver
// detects that the GPU is too hot, it will throttle the clock down below the
// applications clock.  At the moment, this situation is not detected; so keep
// an eye on your temperature with nvidia-smi when running benchmarks!
//
// Note that we don't attempt to set the applications clock ourselves (even
// though it is programatically available) because Kepler and later requires root
// to do so (or for it to have been enabled with 'nvidia-smi -acp UNRESTRICTED'
void check_gpu_applications_clock(unsigned int gpu)
{
  // TODO: If there are multiple NVML invocations, move this out somewhere
  // common / global variable
  NVML_CHECK(nvmlInit());
  nvmlDevice_t gpu_device;
  NVML_CHECK(nvmlDeviceGetHandleByIndex(gpu, &gpu_device));

  unsigned int sm_mhz, mem_mhz, max_sm_mhz, max_mem_mhz;
  NVML_CHECK(nvmlDeviceGetApplicationsClock(gpu_device, NVML_CLOCK_GRAPHICS, &sm_mhz));
  NVML_CHECK(nvmlDeviceGetApplicationsClock(gpu_device, NVML_CLOCK_MEM, &mem_mhz));
  NVML_CHECK(nvmlDeviceGetMaxClockInfo(gpu_device, NVML_CLOCK_GRAPHICS, &max_sm_mhz));
  NVML_CHECK(nvmlDeviceGetMaxClockInfo(gpu_device, NVML_CLOCK_MEM, &max_mem_mhz));
  bool warned = false;
  if (sm_mhz < max_sm_mhz) {
    std::cerr << "WARNING: graphics applications clock speed is " << sm_mhz << "MHz (max is " << max_sm_mhz << "MHz)\n";
    warned = true;
  }
  if (mem_mhz < max_mem_mhz) {
    std::cerr << "WARNING: mem applications clock speed is " << mem_mhz << "MHz (max is " << max_mem_mhz << "MHz)\n";
    warned = true;
  }
  if (warned) {
    std::cerr << "Consider running 'sudo nvidia-smi --applications-clocks=" << max_mem_mhz << "," << max_sm_mhz << "'\n";
    std::cerr << "to maximize the base clock speed and prevent boost clock from introducing variability to benchmark timings.\n";
  }
}

/*
// This is actually not so useful because even if it looks like you're maxing
// the clocks at your sampling rate, actually the GPU may be clocking up/down
// faster than you can observe.  Setting applications-clocks is a lot more
// stable, so that's what the new stuff checks for.
void check_gpu_clock_maxed(unsigned int gpu)
{
  NVML_CHECK(nvmlInit());
  nvmlDevice_t gpu_device;
  NVML_CHECK(nvmlDeviceGetHandleByIndex(gpu, &gpu_device));

  // Check and see if the GPU hit max clock
  unsigned int sm_mhz, mem_mhz, max_sm_mhz, max_mem_mhz;
  NVML_CHECK(nvmlDeviceGetClockInfo(gpu_device, NVML_CLOCK_SM, &sm_mhz));
  NVML_CHECK(nvmlDeviceGetClockInfo(gpu_device, NVML_CLOCK_MEM, &mem_mhz));
  NVML_CHECK(nvmlDeviceGetMaxClockInfo(gpu_device, NVML_CLOCK_SM, &max_sm_mhz));
  NVML_CHECK(nvmlDeviceGetMaxClockInfo(gpu_device, NVML_CLOCK_MEM, &max_mem_mhz));
  bool warned = false;
  if (sm_mhz < max_sm_mhz) {
    std::cerr << "WARNING: After warmup, sm clock speed is " << sm_mhz << "MHz (max is " << max_sm_mhz << "MHz)\n";
    warned = true;
  }
  if (mem_mhz < max_mem_mhz) {
    std::cerr << "WARNING: After warmup, mem clock speed is " << mem_mhz << "MHz (max is " << max_mem_mhz << "MHz)\n";
    warned = true;
  }
  if (warned) {
    std::cerr << "Consider running 'sudo nvidia-smi --applications-clocks=" << max_mem_mhz << "," << max_sm_mhz << "'\n";
    std::cerr << "to max out the GPU clock speed (or figure out why you aren't queueing enough work.)\n";
  }
}
*/

// Modeled off of Soumith's benchmark at
// https://github.com/soumith/convnet-benchmarks/blob/d6177f97e61da0d98a528f355086eb2fc05fe7b8/nervana/convnet-benchmarks.py
int main() {

  // TODO: Check that frequency is fixed

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

  constexpr int iters = fast ? 3 : 200; // truncated backpropagation length
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
    auto start_cpu = getTime();
    for (int j = 0; j < iters; j++) {
      std::tie(hx, cx) = mlstm(input, hx, cx, w_xm, w_hm, w_ih, w_mh);
    }
    auto end_cpu = getTime();
    CUDA_CHECK(cudaEventRecord(end, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    float msecs;
    cudaEventElapsedTime(&msecs, start, end);
    printf("mlstm(%2d): %8.3f msecs (%8.3f msecs cpu)\n", i, msecs, (end_cpu-start_cpu)/1000000.0);
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(end));

  return 0;
}
