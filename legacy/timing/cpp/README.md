How to run

1. Install PyTorch using Anaconda
2. Activate conda environment
3. Build dependencies using build_deps.sh (make sure you checked out all the third_party submodules)
4. From within build/bin run (Replace PYTORCH_HOME with the path to your pytorch repo)
```
cmake ../.. -DPYTORCH_HOME=/scratch/cpuhrsch/repos/pytorch && make -j $(nproc)
```
5. Run benchmarks or add new ones
