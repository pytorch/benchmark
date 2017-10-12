#include <ATen/ATen.h>
#include <iostream>

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

int main() {
  constexpr int batch_size = 3;
  constexpr int input_size = 100;
  constexpr int hidden_size = 400;
  constexpr int embed_size = 400;
  constexpr int n_iters = 700;
  auto input = at::CUDA(at::kFloat).randn({batch_size, input_size});
  auto hx    = at::CUDA(at::kFloat).randn({batch_size, hidden_size});
  auto cx    = at::CUDA(at::kFloat).randn({batch_size, hidden_size});
  auto w_xm  = at::CUDA(at::kFloat).randn({embed_size, input_size});
  auto w_hm  = at::CUDA(at::kFloat).randn({embed_size, hidden_size});
  auto w_ih  = at::CUDA(at::kFloat).randn({4 * hidden_size, input_size});
  auto w_mh  = at::CUDA(at::kFloat).randn({4 * hidden_size, embed_size});
  // TODO: timing
  // TODO: warmup
  for (int i = 0; i < n_iters; i++) {
    std::tie(hx, cx) = mlstm(input, hx, cx, w_xm, w_hm, w_ih, w_mh);
  }
  std::cerr << hx << "\n" << cx << "\n";
  return 0;
}
