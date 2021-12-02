#!/bin/bash

set -euo pipefail

declare -A imagenet

imagenet_base_url=https://image-net.org/data/ILSVRC/2012/

imagenet[val]=ILSVRC2012_img_val
imagenet[train]=ILSVRC2012_img_train
imagenet[test]=ILSVRC2012_img_test_v10102019

if [[ $# < 1 ]]; then
   echo "USAGE : ./getImagenetDataset.sh <val|train|test> [targetFolder]"
   exit 1
fi

SELF_DIR="$( cd "$(dirname "$0")" ; pwd -P )"
MODE="${1}"
TARGET_FOLDER="${2:-${SELF_DIR}/../../data/datasets/imagenet_${MODE}}"

mkdir -p "${TARGET_FOLDER}" && cd "${TARGET_FOLDER}"

if [ -d "${imagenet[${MODE}]}" ]; then
    echo "'${imagenet[${MODE}]}' already exists."
else
    if [ -f "${imagenet[${MODE}]}.tar" ]; then
        echo "'${imagenet[${MODE}]}.tar' already exists."
    else
        URL="${imagenet_base_url}${imagenet[${MODE}]}"
        echo "Downloading dataset: ${imagenet[${MODE}]}.tar from ${URL}"
        wget --no-check-certificate "${URL}"
    fi
    echo "Unpack tar"
    tar -xf ${imagenet[${MODE}]}.tar ${imagenet[${MODE}]}
fi
