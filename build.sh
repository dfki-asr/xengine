#!/bin/bash

mkdir -p build
cd build

mode=0
if [ $# -gt 0 ]
then
   mode=$1
else
   echo "Usage: ./build.sh [0=no solver, 1=CBC, 2=Gurobi, 3=CBC+Gurobi]"
   echo "Default: 0=no solver."
fi

if [ $mode -eq 0 ]
then
   # no Solver
   cmake .. -DDNNL_ROOT_DIR=$ONEDNN_HOME -DONEAPI_ROOT_DIR=$ONEAPI_HOME -DHAS_CBC=OFF -DHAS_GUROBI=OFF
fi

if [ $mode -eq 1 ]
then
   # CBC only
   cmake .. -DDNNL_ROOT_DIR=$ONEDNN_HOME -DONEAPI_ROOT_DIR=$ONEAPI_HOME -DHAS_CBC=ON -DCBC_ROOT_DIR=$CBC_HOME
fi

if [ $mode -eq 2 ]
then
   # GUROBI only
   cmake .. -DDNNL_ROOT_DIR=$ONEDNN_HOME -DONEAPI_ROOT_DIR=$ONEAPI_HOME -DHAS_GUROBI=ON -DGUROBI_ROOT_DIR=$GUROBI_HOME
fi

if [ $mode -eq 3 ]
then
   # CBC and GUROBI
   cmake .. -DDNNL_ROOT_DIR=$ONEDNN_HOME -DONEAPI_ROOT_DIR=$ONEAPI_HOME -DHAS_CBC=ON -DCBC_ROOT_DIR=$CBC_HOME -DHAS_GUROBI=ON -DGUROBI_ROOT_DIR=$GUROBI_HOME
fi

make -j 3
