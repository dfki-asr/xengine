#!/bin/bash

set -euo pipefail

base_url=https://github.com/onnx/models/raw/master/vision/classification

declare -A models
models[vgg16-7]=vgg/model/vgg16-7.onnx
models[vgg19-7]=vgg/model/vgg19-7.onnx
models[resnet18-v1-7]=resnet/model/resnet18-v1-7.onnx
models[resnet50-v1-7]=resnet/model/resnet50-v1-7.onnx
models[resnet34-v1-7]=resnet/model/resnet34-v1-7.onnx

SELF_DIR="$( cd "$(dirname "$0")" ; pwd -P )"

networks=("vgg16-7" "vgg19-7" "resnet18-v1-7" "resnet50-v1-7" "resnet34-v1-7")
default_bs=1

for net in ${networks[@]}; do
   TARGET_FOLDER="${SELF_DIR}/../../data/models"
   cd "${TARGET_FOLDER}"
   URL="${base_url}/${models[${net}]}"
   bs=${default_bs}
   if [ -f "${TARGET_FOLDER}/${net}_bs${bs}.onnx" ]; then
      echo "${TARGET_FOLDER}/${net}_bs${bs}.onnx already exists."
   else
      echo "downloading ${URL} ..."
      wget --no-check-certificate "${URL}"
      echo "save ${net} as ${TARGET_FOLDER}/${net}_bs${bs}.onnx"
      mv "${TARGET_FOLDER}/${net}.onnx" "${TARGET_FOLDER}/${net}_bs${bs}.onnx"
   fi
done
