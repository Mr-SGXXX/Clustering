# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code refers to `https://github.com/facebookresearch/deepcluster`
#
#!/bin/bash

MODELROOT="./weight/deepcluster"

if [ $# -eq 1 ]; then
    MODELROOT=$1
fi

mkdir -p ${MODELROOT}

for MODEL in alexnet vgg16
do
  mkdir -p "${MODELROOT}/${MODEL}"
  for FILE in checkpoint.pth.tar model.caffemodel model.prototxt
  do
    wget -c "https://dl.fbaipublicfiles.com/deepcluster/${MODEL}/${FILE}" \
      -P "${MODELROOT}/${MODEL}" 

  done
done
