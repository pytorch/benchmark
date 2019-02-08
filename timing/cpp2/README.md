How to run cpp2

NB: There are other benchmarks in cpp. Move all of the ones in cpp2 to cpp.

1. Install PyTorch using Anaconda
2. Activate conda environment
3. Build dependencies using build_deps.sh (make sure you checked out all the third_party submodules)
4. From within build run (Replace CMAKE_PREFIX_PATH with your install path)
```
cmake .. -DCMAKE_PREFIX_PATH=<path-to-your-env>/site-packages/torch/share/cmake/ && make -j $(nproc)
```
5. Run benchmarks:
```
./aten_overheads [--benchmark_format=json]
./tensor_shape [--benchmark_format=json]
```
and there are others, check the benchmarks/ folder.
