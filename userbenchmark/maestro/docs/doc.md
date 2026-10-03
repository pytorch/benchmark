# Installation

## Bare-Metal installation

Maestro uses [uv](https://docs.astral.sh/uv/) for dependency management. Install `uv` if you don't have it, then from the `maestro/` directory:

```
uv sync
```

This creates a `.venv/` with all dependencies pinned by `uv.lock`. Activate it with `source .venv/bin/activate`, or prefix commands with `uv run` (e.g. `uv run python src/main.py ...`).

## Docker
Use the docker under deploy/Dockerfile

# Run

Make sure the following environment variables are defined (and equal) on every process:
- `MASTER_ADDR`: the address of one of the hosts/nodes
- `MASTER_PORT`: a random available port


The entrypoint is `main.py`, several commands are available from it. To run the execution of a benchmark, use the `run` command and pass the YAML configuration file, see the API documentation below on how to create this YAML configuration, for example:

```bash
# With torchrun
torchrun --nnodes 1 --nproc-per-node 2 maestro/src/main.py run -c "/path/to/config.yaml"

# With srun
srun -N 1 --gpus-per-node=8 python maestro/src/main.py run -c "/path/to/config.yaml"
```

For more information on the `run` command, run:
```
python main.py run --help
```

Other minor commands are available, for more information run `python main.py --help`:
```bash
python main.py list-blocks # Will list the available blocks
```

# API

## Glossary
- **Block**: A reusable unit of work that enqueues a GPU activity (e.g. NCCL collective, GEMM).
- **Team**: A process group of ranks that participate together in a block.
- **Axis**: A collection of teams created from the same rule; each axis is tied to a dedicated CUDA stream.
- **Operation**: One block scheduled on a specific axis with its parameters.
- **Pattern**: An ordered list of operations plus the axes they rely on.

## Environment variables
- `MAESTRO_LOG_LEVEL`: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).

## Configuration Schema
At the top level the YAML file supports:

```yaml
iters: 30               # benchmark iterations (default 30)
warmup_iters: 20        # warmup iterations  (default 20)
latency_precision: 3    # Defines the precision of the reported latency (default 3)
stop_on_error: false    # Stop execution if one pattern fails (default false)
register_buffers: false # Enable NCCL user buffer registration (default false, see below)
save_results_to: /path/to/results.csv # Must be an absolute path. If provided, will save the results as a CSV, if the path doesn't exist it will be created. Don't forget to mount this path if you use a container
backends:               # optional
  backend_name: <backend_config>
axes: {}                # required - axis definitions (see below)
patterns:               # required - list of patterns
  - <pattern_config>
  - <pattern_config>
  - <pattern_config>
```

A pattern config defines a single pattern, each pattern runs independently one after the other and report performance.
If one pattern fails, the execution will continue on the next one, unless `stop_on_error` is set to True.
The pattern config supports:

```yaml
ops: []   # ordered list of operations
name: ""  # Pattern name
```

See `../examples/` for example configurations.

### Axis definition

Each entry under `axes` maps an axis name to either:

- **Explicit teams**
  ```yaml
  axes:
    tp:
      teams:
        - [0, 1]
        - [2, 3]
  ```

- **Size/stride rule**
  ```yaml
  axes:
    dp:
      team_size: 2
      team_stride: 4
  ```

**Explanation about team_stride and team_size:**
To form an axis, we start by selecting the first rank and adding it to the first team, from there, additional ranks are included by striding a distance of team_stride until team_size ranks are gathered. This forms the first team.
We then move to the next rank not already assigned to a team and repeats the same steps to create subsequent teams. This continues until all ranks are organized into teams, ensuring that each rank belongs to exactly one communicator. If the ranks cannot be divided this way (i.e at least one rank will be in two different teams), an error will be raised.

The parameter `team_size` can be either a number or `max`, in this case the team_size will be `world_size / team_stride`.

>> Note: Currently each axis should attribute a group to every rank.


### Operations definition
Each list item in a pattern's `ops` describes one block execution.
Operation parameters are:
- `block`: Registered block name (see `python/src/blocks`).
- `axis`: Axis key defined in `axes`.
- `name`: Name of the operation (optional, defaults to an automatically generated name)
- Extra fields required by the block family:
  - Communication blocks (`CommBlock` subclasses) expect `vector_size` (supports units `K`, `M`, `G`).
  - GEMM blocks require `mat_a_shape` and `mat_b_shape`, specified as lists and unit-aware.

- **Example of a communication block:**
```yaml
- name: all_gather_stage
  block: torch_all_gather
  axis: dp
  vector_size: 16M    # block-specific parameters
```

> Be careful, some blocks like `MegatronMoeLayerBlock` won't allow overlap of the blocks that are enqueued after him, this is a known issue that may be resolved at some point, for now, please be aware of it and always place blocks like this at the end of the pattern, read the docstring of the block to check this.

### Backends config
Under `backends` you can override runtime implementations by specifying the backend's `name` and the according parameters, here is an exhaustive list of the parameters and their choices:

```yaml
backends:
  axis:
    name: torch     # Axis backend to use, available: [torch]
    backend: nccl   # Communication backend used by the axis backend, the options depend on the backend used, for torch, the options are: [nccl]
  profiler:
    name: cupti     # Profiler to use, available: [cupti]
    output_dir: /tmp/profile_%D   # Optional - Where to store the profiled trace, %D will be replaced by the datetime
```

#### Axes backend
Different backends can be used to handle distributed communication:
- **Torch** (`src/core/axis.py:TorchAxis`): Torch ProcessGroup communication


#### Profiler backend
Select the profiler by configuring `backends.profiler.name`. Each profiler may expect specific environment setup (e.g. CUPTI bindings, GPU visibility) and specific parameters.
- **CUPTI** (`core/profilers/cupti.py`): Captures concurrent kernel traces and optionally writes Chrome trace JSON (`traceEvents`).
- **Torch Profiler** (`core/profilers/torch.py`): Alternate implementation that exports traces through `torch.profiler.profile`, not functional yet.


## Output
The output display performance metrics for each block, the performance metric will depend on the block type, for example GEMM blocks will always have bandwidth=N/A as this is not a relevant metric for them.

In addition to per-block metrics, a "shared bandwidth" metric may be reported at the pattern level when multiple communication blocks execute concurrently. This metric is only computed when all overlapping blocks are communication operations and there is at most one block per axis, ensuring a well-defined parallel execution. The shared bandwidth represents the effective aggregate bandwidth achieved across all concurrent collectives, extending the NCCL notion of bus bandwidth. It is calculated by weighting each collective’s message size by its corresponding bus bandwidth factor (which depends on the communicator size), summing these contributions, and dividing by the maximum execution time among the overlapping blocks. Formally, for N concurrent collectives, the shared bandwidth is defined as:
`SHARED_BW = (s_1 * f_1 + s_2 * f_2 + ... + s_N * f_N) / max(t_1, t_2, ..., t_N)`

Where:
- s_i is the message size of the collective i.
- f_i is the factor of the collectives i (as defined by nccl test), as a function of his PG size.
- t_i is the execution time of the collective i.

Additionally, when there is at most one block per axis, an "overlap percentage" metric is reported. This measures the percentage of the total execution time (from first block start to last block end) during which ALL blocks were running concurrently. A higher overlap percentage indicates better overlap between concurrent operations.

For each block the average, minimum, maximum, and 99th percentile (P99) latencies across iterations are reported. P99 captures tail latency — the value below which 99% of iterations fall — which is useful for surfacing rare slow iterations that the average alone may mask.

Example output (several patterns, only the second one has shared bandwidth and overlap percentage):
```
Pattern: moe+allgather
Block            | Size (B)         | Avg latency (ms) | Min lat. (ms)    | Max lat. (ms)    | P99 lat. (ms)    | Avg BW (GB/s)
---------------------------------------------------------------------------------------------------------------------------------
dp_allgather     | 250M             | 6.0              | 5.6              | 6.2              | 6.2              | 33.0
moe_layer_1      | N/A              | 24.0             | 20.3             | 27.5             | 27.3             | N/A


Pattern: AG+RS
Block            | Size (B)         | Avg latency (ms) | Min lat. (ms)    | Max lat. (ms)    | P99 lat. (ms)    | Avg BW (GB/s)
---------------------------------------------------------------------------------------------------------------------------------
allgather        | 4M               | 0.4              | 0.3              | 0.6              | 0.6              | 9.3
reduce_scatter   | 256M             | 5.6              | 5.5              | 5.7              | 5.7              | 41.9

Avg shared BW (GB/s): 42.6
Avg overlap: 85.3%
```


## Operation presets

Based on hyper parameters, operation presets automatically create the right operations for a pre-configured task, for example a MoE fast forward pass. The operation configuration should have the `preset` key, matching one of the available op presets, a `name` that will be a suffix for every operation created and then a `params` dictionary matching the required parameters of the preset. Run `main.py list-op-presets` to list available presets. The ops presets are located in the `ops_presets` directory, each class has a `params_schema` that describe what parameters are expected. 
