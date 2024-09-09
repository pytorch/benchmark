TORCHBENCH_BRANCH=${TORCHBENCH_BRANCH:-main}

DOCKER_BUILDKIT=0 docker build . --no-cache -f torchbench-nightly.dockerfile -t ghcr.io/pytorch/torchbench:latest \
    --build-arg TORCHBENCH_BRANCH=${TORCHBENCH_BRANCH}
