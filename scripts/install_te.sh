export NVTE_FRAMEWORK=pytorch 
export MAX_JOBS=8
export NVTE_BUILD_THREADS_PER_JOB=1
export CMAKE_BUILD_WITH_INSTALL_RPATH=ON 
export NVTE_CUDA_ARCHS="90"
export CUDA_PATH=/usr/local/cuda

PIP_BREAK_SYSTEM_PACKAGES=1 pip install --no-build-isolation .