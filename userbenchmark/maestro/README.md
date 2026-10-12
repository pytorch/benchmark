# Maestro

Maestro is a benchmarking framework for parallel communication/compute operations on the same GPU.  
This presents a more accurate way to measure performance of AI workloads compared to the micro-benchmarking which measures performance of standalone operations.
It allows a user to define a workload pattern, then benchmark performance of every running block.

For each block, Maestro reports average, minimum, maximum, and 99th percentile (P99) latencies across iterations, plus average bandwidth. P99 latency captures tail behavior that the average can hide — see [docs/doc.md](docs/doc.md#output) for the full output format.

Run from the TorchBench repository root with one process per GPU:

```bash
torchrun --nnodes 1 --nproc-per-node 2 run_benchmark.py maestro /path/to/config.yaml
# In a SLURM allocation:
srun --gpus-per-node=8 --ntasks-per-node=8 python run_benchmark.py maestro /path/to/config.yaml
```

Requires PyTorch 2.9 or newer. Install Maestro dependencies with `python install.py --userbenchmark maestro`.

See the [documentation](docs/) for more:
- [General documentation](docs/doc.md)
- [Developer guide](docs/developer.md)