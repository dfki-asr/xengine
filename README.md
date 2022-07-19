# xengine: Optimal Tensor Rematerialization for Neural Networks in Heterogeneous Environments

XEngine is a Mixed Integer Quadratic Programming approach that schedules network operators onto heterogeneous devices in low memory environments by determining checkpoints and recomputations of tensors for Deep Learning Networks.

Memory efficiency is crucial in training deep learning networks on resource-restricted devices. During backpropagation, forward tensors are used to calculate gradients. Despite the option of keeping those dependencies in memory until they are reused in backpropagation, some forward tensors can be discarded and recomputed later from saved tensors, so-called checkpoints. This allows in particular for resource-constrained heterogeneous environments to make use of all available compute devices. Unfortunately, the definition of these checkpoints is a non-trivial problem and poses a challenge to the programmer, improper or excessive recomputations negate the benefit of checkpointing.

Deep Learning models can be read as .onnx format.
We use Intel oneDNN primitives for the network operators.
Our MIP is written to MPS-fileformat and can be read by common MIP-solvers such as Gurobi and CBC.

## This repository

This repository in related to the Paper "XEngine: Optimal Tensor Rematerialization for Neural Networks in Heterogeneous Environments".

Clone the repository:
```
git clone git@github.com:ManuelaSchuler/xengine.git --recurse-submodules
cd xengine
```

Set the paths to ONEDNN_HOME, GUROBI_HOME, CBC_HOME in prepare.sh
- set ONEDNN_HOME to Install folder of local oneDNN-Installation
- set GUROBI_HOME to [gurobi]/linux64
- set CBC_HOME to [CBC]/dist
and source the prepare.sh file:
```
source prepare.sh
```

Build the project. Select between "0=no Solver", "1=CBC-Only", "2=Gurobi-Only", "3=CBC+Gurobi".

Default: 0=no Solver:
```
./build.sh [0|1|2|3]
```

## Prerequisites and dependencies:

### on the Intel DevCloud

In case you are on the Intel-DevCloud, level-zero and the oneAPI Toolkit are already installed.
You can skip the next two steps and directly jump to the oneDNN-Installation.

### level zero
  This is a dependency of the oneDNN Library.
  Download level-zero from:
  ```
  https://github.com/oneapi-src/level-zero/releases/
  ```
  This repository was tested with level-zero version v1.3-r1.3.0.
  Go to the folder and build and install the level-zero library:
  ```
  mkdir build && cd build && cmake .. && make -j && sudo make install
  ```

### oneAPI Base Toolkit
  Download the oneAPI Toolkit from:
  ```
  https://software.intel.com/content/www/us/en/develop/tools/oneapi/base-toolkit.html
  ```
  Follow the instructions (default installation under: /opt/intel/oneapi) and customize the installation

  ```
  export ONEAPI_HOME=/opt/intel/oneapi
  source $ONEAPI_HOME/setupvars.sh
  ```

### oneDNN Deep Neural Network Library - Version 2.5
  ```
  git clone https://github.com/oneapi-src/oneDNN.git -b rls-v2.5
  cd oneDNN
  mkdir build && cd build
  export CC=icx
  export CXX=icx
  export ONEDNN_HOME=[output folder for oneDNN installation]
  ```
  NOTE: version 2.5 and laster needs the icx compiler instead of clang/clang++ with Ninja

  Build the CPU and GPU runtimes with DPCPP (Intel DPC++ compiler) and icx compiler

  - For CPU + INTEL GPU support
  ```
  cmake .. -DDNNL_CPU_RUNTIME=DPCPP -DDNNL_GPU_RUNTIME=DPCPP
           -DDNNL_GPU_VENDOR=INTEL
           -DCMAKE_INSTALL_PREFIX=$ONEDNN_HOME
           -DDNNL_BUILD_TESTS=OFF
           -DDNNL_BUILD_EXAMPLES=OFF
           -DCMAKE_PREFIX_PATH=/usr/include/level_zero
  ```

  - For CPU + NVIDIA GPU support

  To build for NVIDIA GPUs, CUDA 10.1 has to be installed and the DPCPP Compiler
  has to be build from source (not supported by default):
  ```
  git clone https://github.com/intel/llvm -b sycl
  cd llvm && mkdir build && cd build
  export DPCPP_HOME=$PWD
  python $DPCPP_HOME/llvm/buildbot/configure.py --cuda --no-werror --system-ocl
  python $DPCPP_HOME/llvm/buildbot/compile.py
  ```
  Note: Ensure that "find_package(OpenCL)" finds the oneAPI OpenCL version under $ONEAPI_HOME.
  Otherwise, this will result in compiler errors.

  Then, copy the content of the install folder to oneAPI (bin, include, lib):
  ```
  cp -r install/* $ONEAPI_HOME/compiler/latest/linux
  ```

  Then, build oneDNN:
  ```
  cmake .. -DDNNL_CPU_RUNTIME=DPCPP -DDNNL_GPU_RUNTIME=DPCPP 
           -DDNNL_GPU_VENDOR=NVIDIA
           -DCMAKE_INSTALL_PREFIX=$ONEDNN_HOME
           -DOPENCLROOT=$ONEAPI_HOME/compiler/latest/linux/
           -DOpenCL_INCLUDE_DIR=$ONEAPI_HOME/compiler/latest/linux/include
  ```
  It is important to check that the correct OpenCL files ($ONEAPI_HOME/compiler/latest/include
  and $ONEAPI_HOME/compiler/latest/lib) are used!
  There might be conflicts with system OpenCL headers!
  

  Local changes to oneDNN (release 2.5) - Controlling the Number of CPU Compute Threads:
  
  Apply the following patch in order to control the number of CPU threads via the variable OPENCL_NUM_CORES.
  This step is needed to reproduce results of the paper.

  ```
  diff --git a/src/common/dnnl_thread.cpp b/src/common/dnnl_thread.cpp
index 991dafd8f..c193f219b 100644
--- a/src/common/dnnl_thread.cpp
+++ b/src/common/dnnl_thread.cpp
@@ -31,6 +31,13 @@ namespace impl {
 
 static int adjust_num_threads(int nthr, dim_t work_amount) {
     if (nthr == 0) nthr = dnnl_get_current_num_threads();
+    const char *num_cores = std::getenv("OPENCL_NUM_CORES");
+    if (num_cores != nullptr) {
+        int n_cores = std::stoi(num_cores);
+        if (n_cores < nthr) {
+            nthr = n_cores;
+        }
+    }
 #if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP
     return (work_amount == 1 || omp_in_parallel()) ? 1 : nthr;
 #else
   ```
   
   Finally, build and install oneDNN
   ```
   make -j 4 && make install
   ```

### ILP Solver
  Two ILP Solvers are supported: COIN-OR Cbc solver and the Gurobi Optimization solver.

  Gurobi solver is a commercial solver by the Gurobi company, but the software can be downloaded for free and academic licences can be requested over their webpage.

  Cbc is an open source solver:
  ```
  mkdir CBC && cd CBC
  wget https://raw.githubusercontent.com/coin-or/coinbrew/master/coinbrew
  chmod u+x coinbrew
  ./coinbrew fetch Cbc@2.10.8
  ./coinbrew build Cbc
  ```
  build files will be written to CBC/dist.

  NOTE: Make sure that you use the default gcc or c++ compiler and NOT still the icx compiler (as for oneDNN).

  CBC_ROOT_DIR should be set to the CBC/dist folder.

## Build the xengine project:

```
source $ONEAPI_HOME/setvars.sh
cd xengine
mkdir build
cd build
cmake .. -DDNNL_ROOT_DIR=$ONEDNN_HOME -DONEAPI_ROOT_DIR=$ONEAPI_HOME -DHAS_CBC=ON -DCBC_ROOT_DIR=$CBC_HOME -DHAS_GUROBI=ON -DGUROBI_ROOT_DIR=$GUROBI_HOME
```

## Get mnist dataset
```
cd util/preprocessing
./getMNISTDataset.sh
```
downloads and unzips the mnist train and test dataset into the data/dataset folder.

## Run lenet with xengine

change into your build folder:
```
cd build
```

run lenet with batchsize 64 in inference mode:
```
./examples/xengine lenet 64 0 output_folder [device_file]
```
The "device_file" defines the available devices and in located in folder data/devices.
- devices_cpu.txt for CPU-Only
- devices_cpu_gpu.txt for CPU with one (Intel) GPU
- devices_cpu_gpu_gpu.txt for CPU with two (Intel) GPUs

run lenet with batchsize 64 in inference mode on the CPU:
```
./examples/xengine lenet 64 0 output_folder ../data/devices/devices_cpu.txt
```

run lenet with batchsize 64 in training mode:
```
./examples/xengine lenet 64 1 output_folder [device_file]
```

To run xengine with a higher batchsize:
```
cd util/preprocessing
./createONNXModel.py -m ../../data/models/lenet_bs64.onnx -b 256 -o ../../data/models/lenet_bs256.onnx
```
to create a new ONNX model with batchsize 256 based on the input model.
Make sure to put the new model into the data/models folder und to keep the pattern: modelname_bsXX.onnx

Then, you can run it with:
```
cd ../../build
./examples/xengine lenet 256 0 output_folder [device_file]
./examples/xengine lenet 256 1 output_folder [device_file]
```

## Run the simple "cross engine reorder" (taken from oneDNN examples):

To run it with an Intel GPU:
```
export SYCL_BE=PI_OPENCL
./examples/cross_engine_reorder
```
or
```
export SYCL_BE=PI_LEVEL_ZERO
./examples/cross_engine_reorder
```

To run it with an NVIDIA GPU:
```
export SYCL_BE=PI_CUDA
./examples/cross_engine_reorder
```
