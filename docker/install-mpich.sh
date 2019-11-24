#!/bin/bash
set -ex

pushd /tmp
wget https://www.mpich.org/static/downloads/3.2.1/mpich-3.2.1.tar.gz
tar -zxvf mpich-3.2.1.tar.gz
<<<<<<< HEAD
(cd mpich-3.2.1 && ./configure --prefix=/opt/mpich-3.2.1 --with-pic --enable-static --disable-shared --disable-fortran --disable-mpi-fortran --enable-mpi-thread-mutliple && make -j`nproc` && make install)
=======
(cd mpich-3.2.1 && ./configure --prefix=/opt/mpich-3.2.1 --with-pic --enable-static --disable-shared --disable-fortran --disable-mpi-fortran --enable-mpi-thread-mutliple && make -j`nproc` && make install && make clean)
>>>>>>> 825a5e82... support docker
rm -rf mpich-3.2.1.tar.gz mpich-3.2.1
popd
