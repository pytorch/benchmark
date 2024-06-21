. ${HOME}/miniconda3/etc/profile.d/conda.sh

if [ -z "${CONDA_ENV}" ]; then
  echo "ERROR: CONDA_ENV is not set"
  exit 1
fi

if [ -z "${TEST_CONFIG}" ]; then
  echo "ERROR: TEST_CONFIG is not set"
  exit 1
fi

if [ -z "${SETUP_SCRIPT}" ]; then
  echo "ERROR: SETUP_SCRIPT is not set"
  exit 1
fi

. "${SETUP_SCRIPT}"
conda activate "${CONDA_ENV}"

parent_dir=$(dirname "$(readlink -f "$0")")/../..
cd ${parent_dir}

# Test subprocess worker
if [[ "$TEST_CONFIG" == 'cpu' ]]; then
  python -m torchbenchmark._components.test.test_subprocess
  python -m torchbenchmark._components.test.test_worker
fi

# Test models
python test.py -v -k "$TEST_CONFIG"
