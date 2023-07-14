DEFAULT_PYTHON_VERSION=3.10
CONDA=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
filename=$(basename "$CONDA")
wget "$CONDA"
chmod +x "$filename"
./"$filename" -b -u

. ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate
conda install -y python=${DEFAULT_PYTHON_VERSION}
pip install boto3 pyyaml