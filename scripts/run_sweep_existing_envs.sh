#!/bin/bash
set -e
. ~/miniconda3/etc/profile.d/conda.sh
conda activate base

CONFIG_DIR=""
BENCHMARK_FILTER=""
CONDA_ENVS_DIR="${HOME}/sweep_conda_envs"

print_usage() {
  echo "Usage: run_sweep.sh -c ENVS_FILE -o DATA_OUTPUT_DIR"
}

while getopts 'e:c:fo:p' flag; do
  case "${flag}" in
    c) ENVS_FILE="${OPTARG}";;
    o) DATA_DIR="${OPTARG}";;
    *) print_usage
       exit 1 ;;
  esac
done

if [ -z "${ENVS_FILE}" -o -z "${DATA_DIR}" ];
then
  print_usage
  exit 1
fi

#sudo sh -c "echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo"
sudo nvidia-smi -ac 5001,900
CORE_LIST="24-47"
export GOMP_CPU_AFFINITY="${CORE_LIST}"
export CUDA_VISIBLE_DEVICES=0

echo "Running Benchmarks..."
mkdir -p "${DATA_DIR}"
# for CONFIG_FILE in ${ENVS_FILE};
while read ENV_NAME;
do
    ENV_PATH="${CONDA_ENVS_DIR}/${ENV_NAME}"
    conda activate "${ENV_PATH}"

    #python -c "import torch; print(f'${ENV_NAME}: {torch.__version__}')"

    #pip --version
    #pip install distro py-cpuinfo
    echo "Run benchmark for ${ENV_NAME}"

    taskset -c "${CORE_LIST}" pytest test_bench.py -k "${BENCHMARK_FILTER}" --benchmark-min-rounds 20 --benchmark-json ${DATA_DIR}/$(date +"%Y%m%d_%H%M%S")_${c}.json
    conda deactivate
done < ${ENVS_FILE}

echo "Done"


