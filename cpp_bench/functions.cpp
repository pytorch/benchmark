#include <torch/torch.h>

bool checkSizesStrides(at::Tensor& t) {
  return t.size(1) < 100 && t.stride(1) < 1000;
}
