TORCHBENCH_BRANCH=${TORCHBENCH_BRANCH:-main}

docker build . -f torchbench-nightly.dockerfile -t ghcr.io/pytorch/torchbench:latest \
    --build-arg TORCHBENCH_BRANCH=${TORCHBENCH_BRANCH}
