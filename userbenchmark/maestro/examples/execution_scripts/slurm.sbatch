#!/bin/bash
#SBATCH --output=%x_%j.out
#SBATCH --job-name=pytorch-maestro

# Disclaimer: This sbatch doesn't handle installation

# This is an example SBATCH execution script for maestro using a container
# Refer to Sbatch doc for usage: https://slurm.schedmd.com/sbatch.html


# Read parameters: Paths for config file and results dir (the directory containing config's 'save_results_to' file)
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
        --results-dir)
            if [ $# -lt 2 ]; then
                echo "Missing value for --results-dir" >&2
                exit 1
            fi
            RESULTS_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: sbatch $0 [--config-file PATH] [--results-dir DIR]" >&2
            exit 1
            ;;
    esac
done

# Define environment variables
export NCCL_DEBUG=warn 
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)  \
        MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))


# Print parameters summary
echo "#########################################################"
echo "Timestamp: $(date)"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
echo "Config: $CONFIG_FILE"
echo "Results dir: $RESULTS_DIR"
echo "Log level: $MAESTRO_LOG_LEVEL"
echo "#########################################################"


# Sanity checks
if [ -z "$CONFIG_FILE" ]; then 
    echo "Error: --config-file must be specified." >&2
    echo "Usage: sbatch $0 [--config-file PATH] [--results-dir DIR]" >&2
    exit 1
fi
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: CONFIG_FILE '$CONFIG_FILE' does not exist." >&2
    exit 1
fi

if [ -z "$RESULTS_DIR" ]; then
    echo "Warning: --results-dir was not set, this argument is not mandatory, but if you intend to save your results in a file (i.e used 'save_results_to' option in your config), your results may not be saved without it." >&2
    echo "Usage: sbatch $0 [--config-file PATH] [--results-dir DIR]" >&2
else
    # If it is set, check that it exist
    if [ ! -d "$RESULTS_DIR" ]; then
        echo "Error: RESULTS_DIR '$RESULTS_DIR' does not exist." >&2
        exit 1
    fi
fi


# Run maestro
srun --gpus-per-node=$SLURM_GPUS_PER_NODE \
    --ntasks-per-node=$SLURM_GPUS_PER_NODE \
    bash -c "
    python /PATH/TO/maestro/src/main.py run -c $CONFIG_FILE
    "
