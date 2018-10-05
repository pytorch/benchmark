#include <torch/torch.h>
#include <torch/csrc/jit/interpreter.h>
#include <torch/csrc/jit/symbolic_variable.h>
#include <torch/csrc/jit/custom_operator.h>

#include <iostream>
#include <chrono>

using namespace torch::jit;
using namespace std::chrono;
using Var = torch::jit::SymbolicVariable;

extern bool checkSizesStrides(at::Tensor& t);

template<typename F>
void bench(F f) {
  static constexpr int64_t TIMES = 1000000;
  for (int64_t i = 0; i < 100; ++i) {
    f();
  }
  auto s = high_resolution_clock::now();
  for (volatile int64_t i = 0; i < TIMES; ++i) {
    f();
  }
  auto e = high_resolution_clock::now();
  duration<double, std::nano> d = e - s;
  std::cout << d.count() / TIMES << "ns" << std::endl;
}

RegisterOperators reg({
    Operator(
      "aten::drop(Tensor x, bool y) -> Tensor",
      [](Node * node) {
        return [](Stack& stack) {
          drop(stack, 1);
          return 0;
        };
      }),
    Operator(
      "aten::noop(Tensor x) -> Tensor",
      [](Node * node) {
        return [](Stack& stack) {
          return 0;
        };
      })
});

std::shared_ptr<Graph> id() {
  auto graph = std::make_shared<Graph>();
  auto input = graph->addInput();
  graph->registerOutput(input);
  return graph;
}

std::shared_ptr<Graph> cond() {
  auto graph = std::make_shared<Graph>();
  auto input = graph->addInput();

  auto size = graph->insert(aten::size, {input, 1});
  auto stride = graph->insert(aten::stride, {input, 1});
  auto size_lt = graph->insert(aten::lt, {size, 100});
  auto stride_lt = graph->insert(aten::lt, {stride, 1000});
  auto all = graph->insert(aten::__and__, {size_lt, stride_lt});
  auto result = graph->insert(Symbol::aten("drop"), {input, all});

  graph->registerOutput(result);

  return graph;
}

std::shared_ptr<Graph> noops() {
  auto graph = std::make_shared<Graph>();
  auto input = graph->addInput();

  auto result = input;
  for (int64_t i = 0; i < 100; ++i) {
    result = graph->insert(Symbol::aten("noop"), {result});
  }

  graph->registerOutput(result);

  return graph;
}

int main() {
  auto graph = cond();
  std::cout << *graph << std::endl;

  Stack stack;
  auto t = torch::zeros({10, 10});
  stack.emplace_back(t);

  GraphExecutor executor { graph, false };
  executor.run(stack);

  Code code { graph };

  std::cout << "nothing:         ";
  bench([&]() { });

  std::cout << "one alloc:       ";
  bench([&]() { auto p = new int[18]; delete p; });

  std::cout << "construct IS:    ";
  bench([&]() { InterpreterState asdf(code); });

  std::cout << "construct & run: ";
  bench([&]() { InterpreterState(code).runOneStage(stack); });

  std::cout << "executor run:    ";
  bench([&]() { executor.run(stack); });

  std::cout << "native run:      ";
  bench([&]() { checkSizesStrides(t); });
}
