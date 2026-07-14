# CLAUDE.md - PyTorch Benchmark Repository Guide

This file provides guidance for AI assistants working on the `pytorch-benchmark` repository.

## Repository Overview

This is the **PyTorch Benchmark Suite** — a comprehensive performance testing framework for PyTorch that includes 105+ models across computer vision, NLP, and specialized domains. It is used for continuous performance monitoring, regression detection, and bisection analysis.

## Repository Structure

```
pytorch-benchmark/
├── torchbenchmark/          # Core benchmark library
│   ├── models/              # 105+ benchmark models (each in its own directory)
│   ├── canary_models/       # Experimental/canary models
│   ├── e2e_models/          # End-to-end benchmark models
│   ├── util/                # Core utilities (model loading, profiling, benchmarking)
│   └── _components/         # Component architecture (tasks, worker processes)
├── userbenchmark/           # User-customizable benchmark suites
│   ├── dynamo/              # TorchDynamo compilation testing
│   ├── optim/               # Optimizer performance
│   ├── torch-nightly/       # Nightly PyTorch tracking
│   ├── torchao/             # TorchAO quantization
│   └── ...                  # Other benchmark suites
├── .github/
│   ├── workflows/           # GitHub Actions CI/CD
│   └── scripts/             # Helper scripts (bisection, A/B testing, analysis)
├── .ci/torchbench/          # Local CI scripts (install.sh, test.sh)
├── scripts/                 # Utility scripts (conda, batch sizing, scribe uploads)
├── utils/                   # Build utilities, CUDA, GitHub, S3, version checking
├── docker/                  # Docker configuration for nightly builds
├── submodules/              # External project submodules (FAMBench, lit-llama)
├── test.py                  # Unittest sanity checks for all models
├── test_bench.py            # pytest-benchmark driver for performance measurements
├── run.py                   # CLI for debugging/profiling individual models
├── run_benchmark.py         # Router for userbenchmarks
├── run_e2e.py               # End-to-end model execution
├── install.py               # Main installation orchestrator
├── regression_detector.py   # A/B test result comparison and regression detection
├── bisection.py             # Automated binary search for regression sources
├── conftest.py              # pytest configuration
├── requirements.txt         # Core Python dependencies
├── setup.py                 # Package setup
└── pyproject.toml           # Build system and code style config
```

## Development Workflows

### Installation

```bash
# Install all models
python3 install.py

# Install specific models
python3 install.py --models BERT_pytorch densenet121

# Skip certain models during install
python3 install.py --skip MODEL_NAME

# Install a userbenchmark
python3 install.py --userbenchmark dynamo

# Check if models are installed (no install)
python3 install.py --check-only
```

### Running Benchmarks

```bash
# Unittest sanity checks (accuracy validation)
python3 test.py -k "test_BERT_pytorch_train_cpu"
python3 test.py -k "cuda"   # Run all CUDA tests

# Performance benchmarking with pytest-benchmark
pytest test_bench.py
pytest test_bench.py -k "BERT" --benchmark-autosave
pytest test_bench.py --cpu_only
pytest test_bench.py --cuda_only

# Debug/profile a single model
python3 run.py BERT_pytorch -d cuda -t train
python3 run.py densenet121 -d cpu -t eval --profile
python3 run.py alexnet -d cuda -t eval --bs 32

# End-to-end models
python3 run_e2e.py <model_name> -t eval --bs 16

# Run userbenchmarks
python run_benchmark.py dynamo [benchmark-args]
python run_benchmark.py optim [benchmark-args]
```

### CI Execution

```bash
bash .ci/torchbench/install.sh
bash .ci/torchbench/test.sh
```

### Regression Detection and Bisection

```bash
# Compare two benchmark result sets (A/B test)
python regression_detector.py --control <control.json> --treatment <treatment.json> [--output <result.yaml>]

# Binary search over commits to find regression source
python bisection.py \
  --work-dir <dir> \
  --torch-repos-path <path> \
  --torchbench-repo-path <path> \
  --config <config.yaml> \
  --output <output_file>
```

## Key Conventions

### Model Structure

Every model lives in `torchbenchmark/models/<ModelName>/` and must contain:

```
torchbenchmark/models/MyModel/
├── __init__.py      # Model class implementing BenchmarkModel API
├── install.py       # Model-specific dependency installation
└── metadata.yaml    # Device-specific batch sizes and benchmark config
```

Optional files: `requirements.txt`, `setup.py`, subdirectories for model code.

### BenchmarkModel API

All models inherit from `BenchmarkModel` and implement:

```python
class Model(BenchmarkModel):
    task = TASK_TYPE                        # e.g., NLP.LANGUAGE_MODELING
    DEFAULT_TRAIN_BSIZE = 32               # Default training batch size
    DEFAULT_EVAL_BSIZE = 32                # Default eval batch size
    DEEPCOPY = False                       # Whether to deep copy model between runs
    DISABLE_DETERMINISM = False            # Whether to disable determinism checks

    def __init__(self, test: str, device: str, batch_size: Optional[int] = None, extra_args: List[str] = []):
        ...

    def get_module(self):
        """Return (model, example_inputs)"""
        return self.model, self.example_inputs

    def train(self):
        """Run one training step"""
        ...

    def eval(self):
        """Run one eval step"""
        ...
```

### Userbenchmark Structure

Each userbenchmark in `userbenchmark/<name>/` requires:

```
userbenchmark/my_benchmark/
├── __init__.py       # Required (can be empty)
└── run.py            # Must expose: run(args: List[str])
```

Optional: `install.py`, `regression_detector.py`, `ci.yaml`.

**Output format** (written to `.userbenchmark/<name>/metrics-<timestamp>.json`):
```json
{
  "name": "benchmark-name",
  "environ": {
    "pytorch_git_version": "...",
    "pytorch_version": "..."
  },
  "metrics": {
    "metric_name": value
  }
}
```

### Naming Conventions

- **Classes**: PascalCase (`BenchmarkModel`, `ModelTask`)
- **Functions/Methods**: snake_case (`get_module`, `train`, `eval`)
- **Constants**: UPPERCASE (`DEFAULT_TRAIN_BSIZE`, `DEEPCOPY`)
- **Test names**: `test_<ModelName>_<mode>_<device>` (e.g., `test_BERT_pytorch_train_cpu`)

### Code Style

- **Line length**: 88 characters (Black formatter), 120 characters (flake8 linting)
- **Formatter**: Black (`pyproject.toml` configures this)
- **Linter**: flake8 (see `.flake8` for ignored rules)
- **C++ formatting**: clang-format (`.clang-format`)

### Supported Devices

- `cpu` - CPU execution
- `cuda` - NVIDIA GPU via CUDA
- `mps` - Apple Silicon GPU via Metal Performance Shaders
- `hpu` - Intel Gaudi (Habana)

### Import Patterns

```python
# Core benchmark imports
from torchbenchmark import (
    _list_model_paths,
    ModelTask,
    get_metadata_from_yaml,
)

# Model-specific
from torchbenchmark.util.model import BenchmarkModel
from torchbenchmark.tasks import NLP, COMPUTER_VISION
```

## Adding New Models

Refer to `torchbenchmark/models/ADDING_MODELS.md` for the complete guide. Key steps:

1. Create directory `torchbenchmark/models/<ModelName>/`
2. Implement `__init__.py` with the `BenchmarkModel` subclass
3. Create `install.py` for dependencies
4. Create `metadata.yaml` with device/batch-size configuration
5. Test with `python3 run.py <ModelName> -d cpu -t eval`
6. Run sanity checks: `python3 test.py -k "test_<ModelName>"`

## Adding New Userbenchmarks

Refer to `userbenchmark/ADDING_USERBENCHMARKS.md` for the complete guide. Key steps:

1. Create directory `userbenchmark/<name>/`
2. Add `__init__.py` (can be empty)
3. Implement `run.py` with a `run(args: List[str])` function
4. Optionally add `install.py`, `regression_detector.py`, `ci.yaml`

## CI/CD Overview

### GitHub Actions Workflows

- **`pr-test.yml`**: Main PR workflow
  - Triggers on PRs to main, pushes to main, and manual dispatch
  - Matrix builds: CPU (`linux.24xlarge`) and CUDA (`linux.aws.a100`)
  - Docker-based execution with 240-minute timeout
  - Uses `HUGGING_FACE_HUB_TOKEN` secret for model downloads

- **`build-nightly-docker.yml`**: Builds nightly Docker image
- **`clean-nightly-docker.yml`**: Cleans up old nightly images

### CI Scripts

- `.ci/torchbench/install.sh` — Checks Python version, installs models (with skip list)
- `.ci/torchbench/test.sh` — Executes the test suite

## Dependencies

### Core Requirements (`requirements.txt`)

| Dependency | Purpose |
|------------|---------|
| `transformers==4.57.3` | HuggingFace Transformers (pinned) |
| `timm==1.0.19` | PyTorch Image Models (pinned) |
| `numba>=0.57.0` | JIT compilation for some models |
| `pytest`, `pytest-benchmark` | Testing framework |
| `pandas`, `numpy`, `scipy` | Data analysis |
| `boto3` | AWS S3 integration |
| `nvidia-ml-py>=13.0.0` | NVIDIA GPU monitoring |
| `submitit` | Job submission |
| `pyyaml`, `tabulate` | YAML parsing, output formatting |

### PyTorch Ecosystem (installed separately)

- `torch`, `torchvision`, `torchaudio`

## Metadata YAML Format

Each model's `metadata.yaml` specifies device-specific configs:

```yaml
devices:
  cpu:
    train_batch_size: 4
    eval_batch_size: 8
  cuda:
    train_batch_size: 32
    eval_batch_size: 64
operators:
  - ...
```

## Common Patterns and Pitfalls

### Memory Management

- Models should sync CUDA operations: `torch.cuda.synchronize()` after GPU work
- The test harness checks for CUDA memory leaks between iterations
- Use `DEEPCOPY = True` if model state is mutated during `get_module()` calls

### Determinism

- By default, models are tested with determinism checks for accuracy validation
- Set `DISABLE_DETERMINISM = True` only when a model is inherently non-deterministic (e.g., uses random sampling)

### Batch Size Handling

- Always respect the `batch_size` parameter passed to `__init__`
- Fall back to `DEFAULT_TRAIN_BSIZE` / `DEFAULT_EVAL_BSIZE` if `batch_size=None`
- Device-specific defaults should be in `metadata.yaml`

### Test Filtering

```bash
# Filter by model name
pytest test_bench.py -k "BERT"

# Filter by device
pytest test_bench.py --cuda_only
pytest test_bench.py --cpu_only

# Filter unittest
python3 test.py -k "test_BERT_pytorch_train_cuda"
```

## Useful Files to Reference

| File | Purpose |
|------|---------|
| `torchbenchmark/util/model.py` | `BenchmarkModel` base class definition |
| `torchbenchmark/__init__.py` | Core loading utilities, `ModelTask`, `_list_model_paths` |
| `torchbenchmark/util/env_check.py` | Environment validation helpers |
| `userbenchmark/utils.py` | Shared userbenchmark utilities |
| `conftest.py` | pytest fixtures and configuration |
| `run.py` | Reference for model invocation patterns |
