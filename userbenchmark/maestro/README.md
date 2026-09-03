# Maestro

Maestro is a benchmarking framework for overlapping communication/compute operations in distributed environments.  
This presents a more accurate way to measure performance of AI workloads compared to the micro-benchmarking which measures performance of standalone operations.
It allows a user to define a workload pattern, then benchmark performance of every running block.

For each block, Maestro reports average, minimum, maximum, and 99th percentile (P99) latencies across iterations, plus average bandwidth. P99 latency captures tail behavior that the average can hide — see [docs/doc.md](docs/doc.md#output) for the full output format.

See the [documentation](docs/) for more:
- [Installation & Usage](docs/doc.md)
- [Developer guide](docs/developer.md)