CONDA=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
filename=$(basename "$CONDA")
wget "$CONDA"
chmod +x "$filename"
./"$filename" -b -u

sudo yum makecache --refresh
sudo yum install -y git jq \
                vim wget curl ninja-build cmake \
                gcc kernel-headers kernel-devel kernel-source \
                libglvnd-glx libsndfile

. ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate