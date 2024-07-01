if [ -z "${SETUP_SCRIPT}" ]; then
  echo "ERROR: SETUP_SCRIPT is not set"
  exit 1
fi

. "${SETUP_SCRIPT}"

parent_dir=$(dirname "$(readlink -f "$0")")/../..
cd ${parent_dir}

# Install Tritonbench
python install.py --userbenchmark triton --fbgemm
