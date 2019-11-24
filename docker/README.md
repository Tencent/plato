
# Dockerized development environment

Build docker image

```bash
# this is not essential, we have pushed the docker image to dockerhub.
docker build -t platograph/plato-dev:centos-7 -f Dockerfile.centos-7 .
docker build -t platograph/plato-dev:ubuntu-xenial -f Dockerfile.ubuntu-xenial .
```

Start a new docker container.

```bash
# cd to the project root directory.
cd ..
# start a new container.
docker run -v $(pwd):/data -e USER_NAME=$(id -un) -e USER_ID=$(id -u) -it --rm --name plato_dev platograph/plato-dev:centos-7
# if you want to open anather shell connected to the container.
# docker exec -it -u $(id -u) plato_dev /bin/bash

```

Run the following command in docker.

```bash

# clean previous failed builds.
git clean -dxf
# install 3th party depencies.
./3rdtools.sh install
# test 
BAZEL_LINKOPTS=-static-libstdc++ CC=/opt/mpich-3.2.1/bin/mpicxx LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PWD}/3rd/hadoop2/lib bazel test --test_env=LD_LIBRARY_PATH plato/...
# build examples
BAZEL_LINKOPTS=-static-libstdc++ CC=/opt/mpich-3.2.1/bin/mpicxx bazel build example/...
# mkdir for output
mkdir /tmp/pagerank
# test pagerank
/opt/mpich-3.2.1/bin/mpiexec.hydra -n 4 -hosts localhost -env LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/data/3rd/hadoop2/lib /data/bazel-bin/example/pagerank -input /data/data/graph/v100_e2150_ua_c3.csv -is_directed false -output /tmp/pagerank
# view the output files
ls /tmp/pagerank

```
