%.exe: %.cpp benchmark_common.h
	LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CONDA_PREFIX}/lib \
	g++ -I${CUDA_HOME}/include -I${CONDA_PREFIX}/include \
	    -L${CUDA_HOME}/lib64 -L${CUDA_HOME}/lib64/stubs -L${CONDA_PREFIX}/lib \
	    -lnvidia-ml -lcudart -lATen \
	    -std=c++11 -O3 -Wall -Wextra -pedantic -g \
	    $< -o $@

cudnn_%.exe: cudnn_%.cpp benchmark_common.h
	LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CONDA_PREFIX}/lib:${PYTORCH_HOME}/torch/lib:${CUDNN_LIB_DIR}:${LD_LIBRARY_PATH} \
	g++ -I${CUDA_HOME}/include -I${CONDA_PREFIX}/include \
	    -I${CUDNN_INCLUDE_DIR} \
	    -I${PYTORCH_HOME}/torch/lib/tmp_install/include \
	    -I${PYTORCH_HOME}/torch/lib/tmp_install/include/TH \
	    -I${PYTORCH_HOME}/torch/lib/tmp_install/include/THPP \
	    -I${PYTORCH_HOME}/torch/lib/tmp_install/include/THNN \
	    -I${PYTORCH_HOME}/torch/lib/tmp_install/include/ATen \
	    -L${CUDA_HOME}/lib64 -L${CUDA_HOME}/lib64/stubs \
	    -L${CUDNN_LIB_DIR} -L${CONDA_PREFIX}/lib \
	    -lnvidia-ml -lcudart -lcudnn \
	    -fopenmp \
	    -Wl,${PYTORCH_HOME}/torch/lib/libATen.so.1 \
	    -std=c++11 -O3 -g \
	    $< -o $@

# TODO: I'm not sure why I had to explicitly link against ATen. Seems like
# I'm picking up the wrong one sometimes...
lstm_variable.exe: lstm_variable.cpp benchmark_common.h
	LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CONDA_PREFIX}/lib:${PYTORCH_HOME}/torch/lib:${CUDNN_LIB_DIR}:${LD_LIBRARY_PATH} \
	g++ -I${CUDA_HOME}/include \
	    -I${PYTORCH_HOME}/ \
	    -I${PYTORCH_HOME}/torch/csrc \
	    -I${PYTORCH_HOME}/torch/lib/pybind11/include \
	    -I${PYTORCH_HOME}/torch/lib/tmp_install/include \
	    -I${PYTORCH_HOME}/torch/lib/tmp_install/include/TH \
	    -I${PYTORCH_HOME}/torch/lib/tmp_install/include/THPP \
	    -I${PYTORCH_HOME}/torch/lib/tmp_install/include/THNN \
	    -I${PYTORCH_HOME}/torch/lib/tmp_install/include/ATen \
	    -L${CUDA_HOME}/lib64 -L${CUDA_HOME}/lib64/stubs -L${CONDA_PREFIX}/lib \
	    -lnvidia-ml -lcudart \
	    -fopenmp \
	    -Wl,${PYTORCH_HOME}/torch/lib/libATen.so.1 \
	    -Wl,${PYTORCH_HOME}/torch/_C.cpython-36m-x86_64-linux-gnu.so \
	    -std=c++11 -O3 -g \
	    $< -o $@

run_%: %.exe
	MKL_NUM_THREADS=1 LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CONDA_PREFIX}/lib:${PYTORCH_HOME}/torch/lib:${CUDNN_LIB_DIR}:${LD_LIBRARY_PATH} numactl --cpunodebind=0 --membind=0 taskset 0x1 perf stat ./$<
