```bash
nvcc --keep --dryrun -arch=sm_90 kernel.cu
nvcc -ptx   -arch=sm_90 kernel.cu
nvcc -cubin -arch=sm_90 kernel.cu
nvdisasm kernel.cubin
cuobjdump -sass a.out
```

1. `nvcc --keep --dryrun -arch=sm_90 kernel.cu`
This means run `nvcc` and tell me all the instructions that would have run. This reveals that `nvcc` is not really a compiler, instead it is a "CUDA compiler driver", in the words of Nvidia.

The commands include
* `gcc ... -E -include "cuda_runtime.h" ... "kernel.cu" -o "kernel.cpp4.ii"`: Run the `gcc` compiler, but with the `-E` command, meaning only run the preprocessor
  * `kernel.cpp4.ii`: essentially just the C++ code but with the macros and includes expanded
* `cudafe++ ... "kernel.cpp4.ii"`: There's little documentation about this binary, but it seems to be reponsible from separating host code from device code
  * `kernel.cudafe1.cpp`: preprocessed host code
  * `kernel.module_id`: Something to do with linking host and device
* `gcc -D__CUDA_ARCH__=900 ... "kernel.cu" -o "kernel.cpp1.ii"`: Run the `gcc` preprocessor again, but this time also expand device-specific macros, instead of ignoring them. Resolves stuff like `#ifdef __CUDA_ARCH__`. In principle, not dependent on steps 1-2
  * `kernel.cpp1.ii`: preprocessed host code, with architecture awareness
* `"$CICC_PATH/cicc" ... -arch compute_90 ... "kernel.cpp1.ii" -o "kernel.ptx"`: Real device compiler and optimizer. Compiles C++ into ptx