Example docker workflow

```
sudo docker run --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -i -v `pwd`:/mnt/localdrive --name pytorch_benchmark_cpu_b2cdd08252a089874e4f19aae3feafc7d5b98eb4 -t cpuhrsch/pytorch_benchmark_cpu /bin/bash /mnt/localdrive/install_and_run.sh b2cdd08252a089874e4f19aae3feafc7d5b98eb4
docker commit pytorch_benchmark_cpu_b2cdd08252a089874e4f19aae3feafc7d5b98eb4 pytorch_benchmark_cpu_b2cdd08252a089874e4f19aae3feafc7d5b98eb4
sudo docker run --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -i -v `pwd`:/mnt/localdrive --cpuset-mems 0 --cpuset-cpus 0-19,40-59 -t pytorch_benchmark_cpu_b2cdd08252a089874e4f19aae3feafc7d5b98eb4 /bin/bash -c "cd /mnt/localdrive && /root/miniconda/bin/python run.py"
```


This will create a local image which contains pytorch at revision b2cdd08252a089874e4f19aae3feafc7d5b98eb4.

This will be simplified in an upcoming commit.
