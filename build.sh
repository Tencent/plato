#!/bin/bash

set -ex

ROOT_DIR=$(realpath $(dirname $0))
cd $ROOT_DIR

if [ $1 == "clean" ]; then
    bazel clean
    ./3rdtools.sh distclean
    ./3rdtools.sh install
fi

export CC=${ROOT_DIR}/3rd/mpich/bin/mpicxx
export BAZEL_LINKOPTS=-static-libstdc++
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${ROOT_DIR}/3rd/hadoop2/lib

# test
bazel test plato/...

# build
bazel build example/...
