#include "ATen/ATen.h"
#include <iostream>

using namespace at;

int main() {
  set_num_threads(1);
  auto t = randn(CPU(kFloat), {(int64_t)(1e1)});
  std::cout  << "t: " << t << std::endl;
  for (int64_t i = 0; i < 1; i++) {
    auto& tmp = t.sin_();
    // asm volatile("" : "+m,r"(tmp) : : "memory");
  }
  std::cout  << "t: " << t << std::endl;
}

