## How to add a custom operator benchmark

1. Create a new folder in the `operators` directory.
2. Add an `operator.py` and `__init__.py` file to the new folder.
3. Implement the `Operator` class.
4. Register the operator benchmarks in the `operator.py` file.

### Example

```
operators/
    my_operator/
        __init__.py
        operator.py
```

## `__init__.py`

The `__init__.py` file only needs to import the operator to trigger the registration of the benchmarks.

```
from .operator import Operator
```

## `operator.py`

The `operator.py` file needs to implement the following:

1. `Operator` class: This class should inherit from `BenchmarkOperator`.
2. `get_input_iter`: This method should return an iterator of input examples for the
   operator.
3. `@register_benchmark`: This decorator should be used to register the benchmarks for
   the operator.
4. `get_bwd_fn`: This method should return a callable that performs the backward pass
   for the operator when needed.
5. `get_grad_to_none`: This method should be overridden to set the gradients to your argument for
   the operator when needed.

### Example

```
from torchbenchmark.util.benchmark_registry import register_benchmark
import triton
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)

@triton.jit
def _kernel(XXX):
    # your triton kernel implementation
    pass

def kenrel_wrapper(a, b, activation=""):
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    _kernel[grid](XXX)
    return c

class Operator(BenchmarkOperator):
    def __init__(self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None):
        super().__init__(tb_args, extra_args)
        self.model = Model()

    def get_input_iter(self) -> Generator:
        for i in range(10):
            yield torch.randn(10)

    @register_benchmark(baseline=True)
    def my_operator(self, input) -> Callable:
        return lambda: self.model(input)

    @register_benchmark()
    def my_operator2(self, input) -> Callable:
        return lambda: kernel_wrapper(input)
```
