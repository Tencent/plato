
# Dockerized development environment

Build docker image

```bash
# this is not essential, we have pushed the docker image to dockerhub.
./docker-build.sh
# Or simply build one docker image by
./docker-build.sh centos.7
./docker-build.sh centos.8
./docker-build.sh debian.9
./docker-build.sh debian.10
./docker-build.sh ubuntu.16.04
./docker-build.sh ubuntu.18.04
./docker-build.sh fedora.31
```

How to use docker images.

```bash
# start container under project root directory.
# cd ..
docker run -v $(pwd):/data -e USER_NAME=$(id -un) -e USER_ID=$(id -u) -it --rm platograph/plato-dev:centos.7
# build 3td party libraries.
./3rdtools.sh distclean
./3rdtools.sh install
# build && test.
./build.sh
# run pagerank.
./scripts/run_pagerank_local.sh
```
