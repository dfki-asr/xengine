# xengine

Clone the repository:
```
git clone git@github.com:ManuelaSchuler/xengine.git --recurse-submodules
```

## Prerequisites and dependencies:

### level zero
  ```
  https://github.com/oneapi-src/level-zero/releases/tag/[version]
  mkdir build && cd build && cmake .. && make -j && sudo make install
  ```

### oneAPI Base Toolkit
  DPCPP/C++ Compiler, Intel Math Library
  ```
  https://software.intel.com/content/www/us/en/develop/tools/oneapi/base-toolkit.html
  ```
  Follow the instructions (default installation under: /opt/intel/oneapi) and customize the installation

  ```
  export ONEAPI_HOME=/opt/intel/oneapi
  source $ONEAPI_HOME/setupvars.sh
  ```

### oneDNN Deep Neural Network Library
  ```
  https://github.com/oneapi-src/oneDNN.git -b master
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

### ILP Solver
  Two ILP Solvers are supported: COIN-OR Cbc solver and the Gurobi Optimization solver.

  Gurobi solver is a commercial solver by the Gurobi company, but the software can be downloaded for free and academic licences can be requested over their webpage.

  Cbc is open source:
  ```
  mkdir CBC && cd CBC
  wget https://raw.githubusercontent.com/coin-or/coinbrew/master/coinbrew
  chmod u+x coinbrew
  ./coinbrew fetch Cbc@master
  ./coinbrew build Cbc
  ```
  build files will be written to CBC/dist. CBC_ROOT_DIR should be set to the CBC/dist folder.

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
./examples/xengine lenet 64 0 output_folder
```

run lenet with batchsize 64 in training mode:
```
./examples/xengine lenet 64 1 output_folder
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
./examples/xengine lenet 256 0 output_folder
./examples/xengine lenet 256 1 output_folder
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
