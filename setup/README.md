Dependencies:

Google benchmark
cmake -G "Unix Makefiles" ../../../third_party/benchmark/ -DBENCHMARK_DOWNLOAD_DEPENDENCIES=ON -DCMAKE_BUILD_TYPE=Release
make -j 80


$ sudo apt-get update
$ sudo apt-get install cpuset


Steps taken to guarantee consistent timings
- CPU settings
- - performance governor
- - -sudo cpupower frequency-set --governor performance
- - isolate cpus
- - - use docker cpuset, alternatively isolate entirely using cset or isolcpu
- - numactl to stick to first numa node





