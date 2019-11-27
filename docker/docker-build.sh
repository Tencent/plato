#!/bin/bash

set -ex

DOCKER_DIR=$(realpath $(dirname $0))
ROOT_DIR=$(realpath $DOCKER_DIR/..)
cd $DOCKER_DIR

function build_docker() {
    if [[ $# != 1 ]]; then 
        echo "build_docker should have param: docker tag."
        exit -1;
    fi
    DOCKER_TAG=$1
    DOCKER_FULL_TAG=platograph/plato-dev:$DOCKER_TAG
    docker build -t $DOCKER_FULL_TAG -f Dockerfile.$DOCKER_TAG .
    pushd $ROOT_DIR
    docker run -v $(pwd):/data -e USER_NAME=$(id -un) -e USER_ID=$(id -u) -it --rm $DOCKER_FULL_TAG ./build.sh clean
    popd
}

if [[ x$1 != x ]]; then
    build_docker $1
else
    build_docker centos.7
    build_docker centos.8
    build_docker fedora.31
    build_docker ubuntu.16.04
    build_docker ubuntu.18.04
    build_docker debian.9
    build_docker debian.10
fi
