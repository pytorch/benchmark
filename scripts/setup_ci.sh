#!/usr/bin/env bash
set -ex -o pipefail

# install git lfs and checkout the blob files
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs=2.13.2
git lfs install
git lfs fetch
git lfs checkout .
