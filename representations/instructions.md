```bash
nvcc --keep --dryrun -arch=sm_90 kernel.cu
nvcc -ptx   -arch=sm_90 kernel.cu
nvcc -cubin -arch=sm_90 kernel.cu
nvdisasm kernel.cubin
cuobjdump -sass a.out
```


## NVCC Dryrun  ##
`nvcc --keep --dryrun -arch=sm_90 kernel.cu`
This means run `nvcc` and tell me all the instructions that would have run. This reveals that `nvcc` is not really a compiler, instead it is a "CUDA compiler driver", in the words of Nvidia.


## Compilation Steps ##
https://chatgpt.com/c/69ae3c2b-5bf0-8329-84ba-18191c8145c2

* `gcc ... -E -include "cuda_runtime.h" ... "kernel.cu" -o "kernel.cpp4.ii"`: Run the `gcc` compiler, but with the `-E` command, meaning only run the preprocessor
  * `kernel.cpp4.ii`: essentially just the C++ code but with the macros and includes expanded
* `cudafe++ ... "kernel.cpp4.ii"`: There's little documentation about this binary, but it seems to be reponsible from separating host code from device code
  * `kernel.cudafe1.cpp`: preprocessed host code
  * `kernel.module_id`: Something to do with linking host and device
* `gcc -D__CUDA_ARCH__=900 ... "kernel.cu" -o "kernel.cpp1.ii"`: Run the `gcc` preprocessor again, but this time also expand device-specific macros, instead of ignoring them. Resolves stuff like `#ifdef __CUDA_ARCH__`. In principle, not dependent on steps 1-2
  * `kernel.cpp1.ii`: preprocessed host code, with architecture awareness
* `"$CICC_PATH/cicc" ... -arch compute_90 ... "kernel.cpp1.ii" -o "kernel.ptx"`: Real device compiler and optimizer. Compiles C++ into ptx
  * `kernel.cudafe1.c`: 
  * `kernel.ptx`: 
  * `kernel.cudafe1.stub.c`: Stubs for device kernel functions
  * `kernel.cudafe1.gpu`: 
* `ptxas -arch=sm_90 ... "kernel.ptx" -o "kernel.sm_90.cubin"`: Assembles PTX into real, ELF-format machine code, written in SASS (the name of nvidia assembly)
  * `kernel.sm_90.cubin`: gpu binary
* `fatbinary --create="kernel.fatbin"`: Packages device binaries into a fat (multiarchitecture) binary. This contains only device code, not host code. This is necessary because sometimes device code can have multiple forms of device code (not sure what that means). Also packages in the original PTX, which can be used as fallback when running (idk how or why).
  * `kernel.fatbin.c`
  * `kernel.fatbin`
* `gcc ... "kernel.cudafe1.cpp" -o "kernel.o"`: Compiles host code into an object file. Sanitized from any device code by `cudafe++`
  * `kernel.o`: host code object file
* `nvlink ... "kernel.o" -lcudadevrt -o "a_dlink.sm_90.cubin"`: 
  * `a_dlink.sm_90.cubin`
  * `a_dlink.reg.c`
* `fatbinary --create="a_dlink.fatbin"`: I think this is finally where host and device get packaged together
  * `a_dlink.fatbin.c`:
  * `a_dlink.fatbin`:
