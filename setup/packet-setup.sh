#!/bin/bash

set -x
set -e

export DEBIAN_FRONTEND=noninteractive

# Git
apt-get -q update
apt-get -y -q install git
# Docker
apt-get -y -q install \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
# Check for 0EBF CD88
apt-key fingerprint 0EBFCD88
add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
apt-get -y -q update
apt-get -y -q install docker-ce
# cpuset
apt-get -y -q install cpuset
# Disable turbo
sudo sh -c 'echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo'
sudo sh -c 'echo 100 > /sys/devices/system/cpu/intel_pstate/max_perf_pct'
sudo sh -c 'echo 100 > /sys/devices/system/cpu/intel_pstate/min_perf_pct'
