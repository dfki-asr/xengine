#!/bin/bash

set -euo pipefail

mnist_base_url=http://yann.lecun.com/exdb/mnist/

declare -A mnist
mnist_train=("train-images-idx3-ubyte" "train-labels-idx1-ubyte")
mnist_test=("t10k-images-idx3-ubyte" "t10k-labels-idx1-ubyte")
mnist[train]=mnist_train[@]
mnist[test]=mnist_test[@]

SELF_DIR="$( cd "$(dirname "$0")" ; pwd -P )"

modi=("train" "test")
for mode in ${modi[@]}; do
   TARGET_FOLDER="${SELF_DIR}/../../data/datasets/mnist_${mode}"
   mkdir -p "${TARGET_FOLDER}" && cd "${TARGET_FOLDER}"
   for ds in ${!mnist[${mode}]}; do
      URL="${mnist_base_url}${ds}"
      if [ -f "${TARGET_FOLDER}/${ds}" ]; then
         echo "${TARGET_FOLDER}/${ds} already exists."
      else
         if [ -f "${TARGET_FOLDER}/${ds}.gz" ]; then
            echo "unzip ${TARGET_FOLDER}/${ds}.gz ..."
            gzip -d "${TARGET_FOLDER}/${ds}.gz"
         else
            echo "download ${URL}.gz and unzip afterwards ..."
            wget --no-check-certificate "${URL}.gz"
            gzip -d "${TARGET_FOLDER}/${ds}.gz"
         fi
      fi
   done
done
