. ${HOME}/miniconda3/etc/profile.d/conda.sh

if [ -z "${CONDA_ENV}" ]; then
  echo "ERROR: CONDA_ENV is not set"
  exit 1
fi

conda activate ${CONDA_ENV}

parent_dir=$(dirname "$(readlink -f "$0")")/..
cd ${parent_dir}

# Test subprocess worker
python -m components.test.test_subprocess
python -m components.test.test_worker

# Test models
python test.py