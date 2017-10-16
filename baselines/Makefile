%.exe: %.cpp benchmark_common.h
	LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CONDA_PREFIX}/lib \
	g++ -I${CUDA_HOME}/include -I${CONDA_PREFIX}/include \
	    -L${CUDA_HOME}/lib64 -L${CUDA_HOME}/lib64/stubs -L${CONDA_PREFIX}/lib \
	    -lnvidia-ml -lcudart -lATen \
	    -std=c++11 -O3 -Wall -Wextra -pedantic -g \
	    $< -o $@

run_%: %.exe
	MKL_NUM_THREADS=1 LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CONDA_PREFIX}/lib numactl --cpunodebind=0 --membind=0 ./$<
