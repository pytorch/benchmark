#!/bin/bash
set -e
. ~/miniconda3/etc/profile.d/conda.sh
conda activate base

TORCH_VER=(
    # "1.4.0" \
    "1.5.1" \
    "1.6.0" \
    "1.7.0.dev20200925"
    )
TORCH_INSTALL=(
    # "conda install -q -y pytorch==1.4.0 torchvision==0.5.0 torchtext==0.4.0 cudatoolkit=10.1 -c pytorch" \
    "conda install -q -y pytorch==1.5.1 torchvision==0.6.1 torchtext==0.5.0 cudatoolkit=10.2 -c pytorch" \
    "conda install -q -y pytorch==1.6.0 torchvision==0.7.0 torchtext==0.7.0 cudatoolkit=10.2 -c pytorch" \
    "conda install -q -y pytorch=1.7.0.dev20200925 torchvision=0.8.0.dev20200925 torchtext==0.8.0.dev20200925 -c pytorch-nightly"
    )
BENCHMARK=benchmark

# for i in ${!TORCH_VER[@]};
# do
#     echo "Install torch ${TORCH_VER[$i]} with cmd: ${TORCH_INSTALL[$i]}"
#     conda create -y -p torch-${TORCH_VER[$i]}-env python=3.7
#     conda activate `pwd`/torch-${TORCH_VER[$i]}-env 
#     ${TORCH_INSTALL[$i]}
#     if [ ${TORCH_VER[$i]} != `python -c 'import torch; print(torch.__version__)'` ]
#     then
#         echo "Failed to install torch ${TORCH_VER}"
#         exit 1
#     fi

#     echo "Install benchmark in ${TORCH_VER[$i]}"
#     pushd ${BENCHMARK}
#     python install.py

#     echo "Test benchmark in ${TORCH_VER[$i]}"
#     python test.py
#     echo "Test passed for ${TORCH_VER[$i]}"
#     popd # BENCHMARK
#     conda deactivate
# done
# echo "Done"

for i in ${!TORCH_VER[@]};
do
    conda activate `pwd`/torch-${TORCH_VER[$i]}-env 
    pushd ${BENCHMARK}
    pytest test_bench.py -k "(not slomo)" --benchmark-min-rounds 20 --benchmark-json .data/benchmark_${TORCH_VER[$i]}_$(date +"%Y%m%d_%H%M%S").json
    popd #BENCHMARK
    conda deactivate
done



