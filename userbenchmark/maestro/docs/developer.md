# Maestro Developer Guide

This guide provides comprehensive documentation for developers looking to extend, modify, or contribute to the Maestro benchmarking framework.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Core Concepts](#core-concepts)
  - [Blocks](#blocks)
  - [Axes](#axes)
  - [Patterns](#patterns)
  - [Orchestrator](#orchestrator)
  - [Profilers](#profilers)
- [Extending Maestro](#extending-maestro)
  - [Adding a New Block](#adding-a-new-block)
  - [Adding a New Profiler](#adding-a-new-profiler)
  - [Adding a New Operation Preset](#adding-a-new-operation-preset)
  - [Adding a New Axis Backend](#adding-a-new-axis-backend)
- [Configuration Parsing](#configuration-parsing)
- [Testing and Debugging](#testing-and-debugging)
- [Code Style and Conventions](#code-style-and-conventions)

---

## Architecture Overview

Maestro follows a modular architecture designed to benchmark parallel communication and compute operations on GPUs. The key principle is measuring performance of concurrent operations (as seen in real AI workloads) rather than isolated micro-benchmarks.

```
┌─────────────────────────────────────────────────────────────┐
│                        main.py (CLI)                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Config Parser                           │
│              (YAML → Internal Configuration)                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       Orchestrator                          │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐ │
│  │  Axes   │  │ Blocks  │  │ Tensor  │  │    Profiler     │ │
│  │         │  │         │  │  Pool   │  │                 │ │
│  └─────────┘  └─────────┘  └─────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     GPU Execution                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │  Stream 1   │  │  Stream 2   │  │  Stream N   │          │
│  │  (Axis 1)   │  │  (Axis 2)   │  │  (Axis N)   │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

**Data Flow:**
1. User provides a YAML configuration file
2. `config_parser.py` parses and validates the configuration
3. `Orchestrator` instantiates axes, blocks, and manages tensor pools
4. Blocks execute on their assigned axes (CUDA streams)
5. Profiler captures timing information
6. Results are aggregated and reported

---

## Project Structure

```
maestro/
├── src/
│   ├── main.py                 # CLI entrypoint
│   ├── blocks/                 # Block implementations
│   │   ├── __init__.py
│   │   ├── torch/              # PyTorch-based blocks
│   │   │   ├── comm.py         # Communication blocks (AllReduce, AllGather, etc.)
│   │   │   └── gemm.py         # GEMM compute block
│   │   └── megatron/           # Megatron-specific blocks
│   │       └── moe_layer.py
│   ├── core/                   # Core framework
│   │   ├── axis.py             # Axis abstraction
│   │   ├── block.py            # Block base classes and registry
│   │   ├── config_parser.py    # Configuration parsing
│   │   ├── op_preset.py        # Operation preset base class
│   │   ├── orchestrator.py     # Main execution orchestrator
│   │   ├── profilers/          # Profiler implementations
│   │   │   ├── __init__.py
│   │   │   ├── base.py         # Profiler base class
│   │   │   ├── cupti.py        # CUPTI-based profiler
│   │   │   └── torch.py        # PyTorch profiler
│   │   └── utils/              # Utility modules
│   │       ├── distributed.py  # Distributed utilities
│   │       ├── format.py       # Size formatting utilities
│   │       └── logging.py      # Logging configuration
│   └── ops_presets/            # Operation preset implementations
│       ├── __init__.py
│       └── moe.py              # MoE operation preset
├── docs/                       # Documentation
├── examples/                   # Example configurations
├── deploy/                     # Deployment scripts
└── pyproject.toml
```

---

## Core Concepts

### Blocks

A **Block** is the fundamental unit of work in Maestro. Each block represents a discrete GPU operation such as an NCCL collective or a GEMM computation.

#### Block Hierarchy

```
_Block (ABC)
├── CommBlock           # Communication operations
│   └── _TorchCommBlock
│       ├── TorchAllReduce
│       ├── TorchAllGather
│       ├── TorchReduceScatter
│       ├── TorchAllToAll
│       └── ...
├── ComputeBlock        # Compute operations
│   └── GEMMBlock
│       └── TorchGEMM
├── CopyBlock           # Memory copy operations
└── TorchNNModuleBlock  # torch.nn.Module wrappers
    └── MegatronModuleBlock
```

#### Block Registry

Blocks are automatically registered via the `BlockRegistry` when they define a `registry_name` class attribute:

```python
class BlockRegistry:
    _registry = {}
    
    @classmethod
    def register(cls, block: type):
        registry_name = getattr(block, "registry_name", None)
        if registry_name:
            cls._registry[registry_name] = block
    
    @classmethod
    def get_block_cls(cls, name: str) -> type:
        return cls._registry[name]
```

The registration happens automatically through the `_BlockRegistrer` metaclass when a block class is defined.

#### Key Block Methods

| Method | Description |
|--------|-------------|
| `run()` | Execute the GPU operation |
| `get_tensor_size(api_params)` | Return required tensor size for this block |
| `from_api_params(axis, name, api_params, cached_tensor)` | Factory method to create block from API parameters |
| `get_block_size_str()` | Human-readable size string for output |

---

### Axes

An **Axis** represents a collection of teams (process groups) that execute operations together. Each axis is bound to a dedicated CUDA stream, enabling concurrent execution across different axes.

#### Axis Configuration

Axes can be defined in two ways:

1. **Explicit teams:**
   ```yaml
   axes:
     tp:
       teams:
         - [0, 1]
         - [2, 3]
   ```

2. **Size/stride rule:**
   ```yaml
   axes:
     dp:
       team_size: 2
       team_stride: 4
   ```

The stride rule works by:
1. Starting with rank 0, add ranks at intervals of `team_stride` until `team_size` is reached
2. Move to the next unassigned rank and repeat

#### TorchAxis Implementation

The `TorchAxis` class provides the PyTorch/NCCL implementation:

```python
class TorchAxis(Axis):
    dist_utils = nccl_torch_dist_utils

    def __init__(self, groups: list[list[int]], name: str = ""):
        super().__init__(groups, name=name)
        self.stream = torch.cuda.Stream()  # Dedicated stream
    
    def _axis_ctx(self):
        return torch.cuda.stream(self.stream)
```

---

### Patterns

A **Pattern** is an ordered sequence of operations that represents a workload to benchmark. Each pattern runs independently and reports performance metrics.

```yaml
patterns:
  - name: "my_pattern"
    ops:
      - block: "torch_all_gather"
        axis: dp
        vector_size: 16M
      - block: "torch_gemm"
        axis: dp
        mat_a_shape: [1024, 4096]
        mat_b_shape: [4096, 1024]
```

---

### Orchestrator

The **Orchestrator** (`core/orchestrator.py`) is the central coordinator that:

1. **Instantiates axes** from configuration
2. **Creates blocks** from pattern definitions
3. **Manages tensor pools** for memory efficiency
4. **Executes patterns** with warmup and measurement iterations
5. **Matches streams to blocks** for accurate profiling
6. **Aggregates results** across ranks and iterations
7. **Reports performance** metrics

#### Execution Flow

```python
def run_pattern(self, pattern: dict):
    # 1. Create blocks from pattern ops
    blocks = self._create_blocks(pattern)
    
    # 2. Warmup (independent of measurement warmup)
    for _ in range(20):
        self._execute(blocks)
    
    # 3. Match streams to blocks
    streams_and_activities_per_block = self._match_streams_to_blocks(blocks)
    
    # 4. Benchmark with profiler
    with self.profiler:
        self._execute(blocks, iters=self.warmup_iters)
        self.profiler.reset()
        self._execute(blocks, iters=self.iters)
    
    # 5. Measure and report
    results = self.profiler.get_results()
    return self._measure_blocks_performance(results, blocks, ...)
```

---

### Profilers

**Profilers** capture GPU activity timing information. They must inherit from `profiler_ctx` and implement the context manager protocol.

#### Profiler Interface

```python
class profiler_ctx(ABC):
    def __init__(self, output_dir: Optional[Path] = None):
        self._results = []
        self.output_dir = output_dir
    
    @abstractmethod
    def __enter__(self):
        pass
    
    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        # Save results to self._results
        pass
    
    def get_results(self) -> list[dict]:
        return self._results
    
    def reset(self):
        self._results = []
```

#### Result Format

Each result dictionary must contain:
- `stream_id`: CUDA stream identifier
- `duration_ns`: Duration in nanoseconds
- `kind`: Activity type ("kernel", "memcpy", etc.)
- `start_ns`: Start timestamp
- `end_ns`: End timestamp

---

## Extending Maestro

### Adding a New Block

#### Step 1: Choose the Base Class

Select the appropriate base class for your block:

| Base Class | Use Case |
|------------|----------|
| `CommBlock` | Communication operations (collectives) |
| `GEMMBlock` | Matrix multiplication operations |
| `ComputeBlock` | Generic compute operations |
| `TorchNNModuleBlock` | Wrapping `torch.nn.Module` |

#### Step 2: Implement the Block

Create your block in the appropriate directory under `src/blocks/`:

```python
# src/blocks/torch/my_block.py
from core.block import CommBlock
from core.axis import TorchAxis

class MyCustomBlock(CommBlock):
    """Custom communication block"""
    
    registry_name = "my_custom_block"  # Used in YAML configs
    
    api_params_schema = {
        "vector_size": {},  # Required parameter
        "custom_param": {"default": 10},  # Optional with default
    }
    
    def __init__(self, axis: TorchAxis, src_buf, dst_buf, custom_param: int, name: str = ""):
        super().__init__(axis, src_buf, dst_buf, name)
        self.custom_param = custom_param
    
    def run(self):
        """Execute the GPU operation"""
        with self.axis.use_axis():
            # Your CUDA operation here
            pass
    
    @classmethod
    def get_tensor_size(cls, api_params: dict) -> int:
        """Return total tensor size needed"""
        return api_params["vector_size"]
    
    @classmethod
    def from_api_params(cls, axis, name: str, api_params: dict, cached_tensor=None):
        """Factory method to create block from config"""
        api_params = cls._enrich_api_params_schema(api_params)
        
        if cached_tensor is None:
            cached_tensor = torch.zeros(cls.get_tensor_size(api_params), device="cuda")
        
        src_buf = cached_tensor[:api_params["vector_size"]]
        dst_buf = cached_tensor[:api_params["vector_size"]]
        
        return cls(
            axis=axis,
            src_buf=src_buf,
            dst_buf=dst_buf,
            custom_param=api_params["custom_param"],
            name=name
        )
    
    @staticmethod
    def get_bus_bw_factor(team_size: int):
        """Bus bandwidth factor for BW calculation"""
        return (team_size - 1) / team_size
```

#### Step 3: Register the Block

Import your block in the package's `__init__.py`:

```python
# src/blocks/torch/__init__.py
from .comm import *
from .gemm import *
from .my_block import MyCustomBlock  # Add this line
```

#### Step 4: Use in Configuration

```yaml
patterns:
  - ops:
    - block: "my_custom_block"
      axis: dp
      vector_size: 16M
      custom_param: 20
```

---

### Adding a New Profiler

#### Step 1: Create the Profiler Class

```python
# src/core/profilers/my_profiler.py
from core.profilers.base import profiler_ctx
from pathlib import Path
from typing import Optional

class MyProfiler(profiler_ctx):
    """Custom profiler implementation"""
    
    def __init__(self, output_dir: Optional[Path] = None):
        super().__init__(output_dir)
        # Initialize profiler-specific state
    
    def __enter__(self):
        """Start profiling"""
        # Enable profiling
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Stop profiling and collect results"""
        results = []
        
        # Collect profiling data
        for activity in self._get_activities():
            results.append({
                "kind": activity.kind,
                "stream_id": activity.stream_id,
                "start_ns": activity.start,
                "end_ns": activity.end,
                "duration_ns": activity.end - activity.start,
            })
        
        self._save_results(results)
        self.save_trace()
    
    def _save_trace(self, output_file: Path):
        """Optional: Save trace file for visualization"""
        # Implement trace export
        pass
```

#### Step 2: Register in main.py

```python
# src/main.py
profiler_backend = config["backends"]["profiler"]["name"]
if profiler_backend == "cupti":
    from core.profilers import CuptiProfiler 
    profiler = CuptiProfiler(output_dir=config["backends"]["profiler"]["output_dir"])
elif profiler_backend == "my_profiler":  # Add this
    from core.profilers.my_profiler import MyProfiler
    profiler = MyProfiler(output_dir=config["backends"]["profiler"]["output_dir"])
else:
    raise ValueError(f"Invalid profiler backend: {profiler_backend}")
```

#### Step 3: Update __init__.py

```python
# src/core/profilers/__init__.py
from .base import profiler_ctx
from .cupti import CuptiProfiler
from .my_profiler import MyProfiler  # Add this
```

---

### Adding a New Operation Preset

Operation presets auto-generate operations based on high-level parameters (e.g., MoE layer configuration).

#### Step 1: Create the Preset Class

```python
# src/ops_presets/my_preset.py
from core.op_preset import OpPreset

class MyOpPreset(OpPreset):
    """Custom operation preset"""
    
    name = "my_preset"  # Used in YAML configs
    
    params_schema = {
        "axis_name": {
            "required": True,
            "description": "Axis to use for operations",
        },
        "size": {
            "required": True,
            "description": "Data size in bytes",
        },
        "optional_param": {
            "default": 42,
            "description": "An optional parameter",
        },
    }
    
    @classmethod
    def create_op(cls, params: dict, axes_config: dict) -> tuple[dict, list]:
        """
        Generate axes configuration and operations list.
        
        Args:
            params: Parameters from YAML configuration
            axes_config: Existing axes configuration
        
        Returns:
            tuple: (new_axes_config, ops_list)
        """
        cls.validate_params(params)
        
        axis_name = params["axis_name"]
        size = params["size"]
        
        # Validate axis exists
        if axis_name not in axes_config:
            raise ValueError(f"Axis {axis_name} not defined")
        
        # Generate operations
        ops = [
            {
                "block": "torch_all_gather",
                "axis": axis_name,
                "name": "phase_1",
                "vector_size": size,
            },
            {
                "block": "torch_gemm",
                "axis": axis_name,
                "name": "phase_2",
                "mat_a_shape": [1024, 4096],
                "mat_b_shape": [4096, 1024],
            },
        ]
        
        # Return empty dict if no new axes needed
        new_axes = {}
        return new_axes, ops
```

#### Step 2: Register the Preset

```python
# src/ops_presets/__init__.py
from .moe import MOEOpPreset
from .my_preset import MyOpPreset  # Add this
```

#### Step 3: Use in Configuration

```yaml
patterns:
  - name: "preset_example"
    ops:
      - preset: "my_preset"
        name: "custom_workload"
        params:
          axis_name: dp
          size: 16000000
```

---

### Adding a New Axis Backend

Axis backends must implement the `Axis` interface.

#### Step 1: Create the Axis Class

```python
# src/core/axis.py (or separate file)
from core.axis import Axis
from your_backend import dist_utils as my_dist_utils

class MyBackendAxis(Axis):
    """Custom axis backend"""
    
    dist_utils = my_dist_utils  # Your distributed utilities
    
    def __init__(self, groups: list[list[int]], name: str = ""):
        super().__init__(groups, name=name)
        # Initialize backend-specific resources
        self.stream = create_stream()  # Your stream creation
    
    @staticmethod
    def create_event():
        """Create synchronization event"""
        return MyEvent()
    
    @staticmethod
    def record_event(event):
        """Record event on current stream"""
        event.record()
    
    def wait_event(self, event):
        """Wait for event on this axis's stream"""
        self.stream.wait_event(event)
    
    def _axis_ctx(self):
        """Context manager for using this axis"""
        return stream_context(self.stream)
    
    def synchronize(self):
        """Synchronize this axis's stream"""
        self.stream.synchronize()
    
    @classmethod
    def synchronize_all(cls):
        """Synchronize all GPU streams"""
        device_synchronize()
    
    @classmethod
    def destroy(cls):
        """Cleanup resources"""
        cls.dist_utils.destroy()
```

#### Step 2: Register in main.py

```python
# src/main.py
axis_backend = config["backends"]["axis"]["name"]
if axis_backend == "torch":
    axis_cls = TorchAxis
elif axis_backend == "my_backend":  # Add this
    from core.axis import MyBackendAxis
    axis_cls = MyBackendAxis
else:
    raise ValueError(f"Invalid axis backend: {axis_backend}")
```

---

## Configuration Parsing

The `config_parser.py` module handles YAML configuration parsing and validation.

### Parse Flow

```
YAML File → parse_config() → Internal Config Dict
                  │
                  ├── parse_axis_config()    → Axis teams
                  ├── parse_pattern()        → Pattern with ops
                  │       └── parse_op_preset()  → Unwrap presets
                  └── parse_backends()       → Backend config
```

### Key Functions

| Function | Description |
|----------|-------------|
| `parse_config(config)` | Main entry point, returns validated config dict |
| `parse_axis_config(axis_config)` | Convert axis config to teams list |
| `parse_pattern(pattern, axes)` | Process pattern, unwrap presets, convert sizes |
| `parse_op_preset(op, axes)` | Expand operation preset into operations |
| `parse_backends(backends)` | Apply backend defaults |

### Size Parsing

The `parse_size()` utility supports human-readable sizes:

```python
"16M"  → 16 * 1024 * 1024
"4G"   → 4 * 1024 * 1024 * 1024
"256K" → 256 * 1024
```

---

## Testing and Debugging

### Logging

Set the log level via environment variable:

```bash
export MAESTRO_LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### Listing Available Components

```bash
# List all registered blocks
python main.py list-blocks

# List all operation presets
python main.py list-op-presets
```

### Debugging Tips

1. **Block not registering**: Ensure `registry_name` is defined and the module is imported in `__init__.py`

2. **Stream mismatch errors**: Check that your block's `run()` method uses `with self.axis.use_axis():`

3. **Profiler not capturing activities**: Verify the profiler is capturing the correct activity types for your block

4. **Tensor size errors**: Check `get_tensor_size()` returns the correct size for your block

### Example Debug Configuration

```yaml
iters: 1
warmup_iters: 1
stop_on_error: true  # Stop on first error
backends:
  profiler:
    name: cupti
    output_dir: "/tmp/maestro_debug"  # Save traces
axes:
  dp:
    team_size: 2
    team_stride: 1
patterns:
  - name: "debug_pattern"
    ops:
      - block: "torch_all_reduce"
        axis: dp
        vector_size: 1M
```

---

## Code Style and Conventions

### General Guidelines

- Use type hints for function signatures
- Follow PEP 8 style guide
- Use descriptive variable names
- Add docstrings for public methods

### Block Naming

- Registry names should be lowercase with underscores: `torch_all_gather`
- Class names should be PascalCase: `TorchAllGather`

### Logging

Use the provided loggers:

```python
from core.utils.logging import get_logger, get_root_rank_logger

logger = get_logger(__name__)        # All ranks
rlogger = get_root_rank_logger()     # Root rank only

logger.debug("Debug message")
rlogger.info("Info from root only")
```

### Error Handling

- Raise `ValueError` for configuration/parameter errors
- Raise `RuntimeError` for execution errors
- Include helpful error messages with context

---

## Quick Reference

### Adding a Communication Block

1. Inherit from `CommBlock`
2. Set `registry_name`
3. Implement `run()` with `with self.axis.use_axis():`
4. Override `get_bus_bw_factor()` for bandwidth calculation
5. Import in `blocks/__init__.py`

### Adding a Compute Block

1. Inherit from `GEMMBlock` or `ComputeBlock`
2. Set `registry_name`
3. Define `api_params_schema`
4. Implement `run()`, `get_tensor_size()`, `from_api_params()`
5. Import in `blocks/__init__.py`

### Adding a Profiler

1. Inherit from `profiler_ctx`
2. Implement `__enter__()` and `__exit__()`
3. Save results with mandatory fields: `stream_id`, `duration_ns`, `kind`
4. Register in `main.py`

### Adding an Operation Preset

1. Inherit from `OpPreset`
2. Set `name` and `params_schema`
3. Implement `create_op()` returning `(axes_dict, ops_list)`
4. Import in `ops_presets/__init__.py`
