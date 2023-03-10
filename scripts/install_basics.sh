CONDA=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
filename=$(basename "$CONDA")
wget "$CONDA"
chmod +x "$filename"
./"$filename" -b -u

sudo apt-get -y update
sudo apt-get install -y git jq \
                vim wget curl ninja-build cmake \
                libgl1-mesa-glx libsndfile1-dev

# Install gcc-11, needed by the latest fbgemm
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    sudo apt install -y g++-11
