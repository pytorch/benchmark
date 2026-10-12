#!/bin/bash
#SBATCH --output=%x_%j.out
#SBATCH --job-name=pytorch-maestro

# Run from the TorchBench repository root after installing dependencies.
# Example: sbatch --nodes=2 --gpus-per-node=8 --ntasks-per-node=8 \
#   userbenchmark/maestro/examples/execution_scripts/slurm.sbatch --config-file /absolute/config.yaml
set -euo pipefail

CONFIG_FILE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --config-file)
            if [ $# -lt 2 ]; then
                echo "Missing value for --config-file" >&2
                exit 1
            fi
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            echo "Usage: sbatch $0 --config-file PATH" >&2
            exit 1
            ;;
    esac
done

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: --config-file must name an existing YAML file." >&2
    exit 1
fi
CONFIG_FILE=$(realpath "$CONFIG_FILE")
cd "${SLURM_SUBMIT_DIR:?Run sbatch from the TorchBench repository root}"
if [ ! -f run_benchmark.py ]; then
    echo "Error: run sbatch from the TorchBench repository root." >&2
    exit 1
fi

# SLURM_GPUS_PER_NODE can contain a GPU type (for example a100:8).
GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-}}
GPUS_PER_NODE=${GPUS_PER_NODE##*:}
if ! [[ "$GPUS_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: specify --gpus-per-node with a positive GPU count." >&2
    exit 1
fi
export NCCL_DEBUG=warn
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))
export MASTER_ADDR MASTER_PORT

# Results are written to the absolute save_results_to path in the YAML.
srun --gpus-per-node="$GPUS_PER_NODE" --ntasks-per-node="$GPUS_PER_NODE" \
    python run_benchmark.py maestro "$CONFIG_FILE"
