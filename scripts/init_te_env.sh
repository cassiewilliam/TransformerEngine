export NVTE_FRAMEWORK=pytorch 
export MAX_JOBS=8
export NVTE_BUILD_THREADS_PER_JOB=1
export CMAKE_BUILD_WITH_INSTALL_RPATH=ON 
export NVTE_CUDA_ARCHS="90a"
export CUDA_PATH=/usr/local/cuda

python3 setup.py build_ext --inplace