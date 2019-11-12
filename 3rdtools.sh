#!/bin/bash

me=`basename "$0"`
rootdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
sourceroot="${rootdir}/.downloads"

function clean_exec {
  cmd=$*
  eval $cmd
  retcode=$?
  if [[ $retcode != 0 ]]; then
    echo "'${cmd}' exec failed with code $retcode, abort install process!"
    exit 255
  fi
}

function show_help {
  echo "usage: ${me} <install|distclean>"
}

function install {
  # create temporary dir to hold source code
  mkdir -p ${sourceroot}
  pushd ${sourceroot}

  ## boost
  if [ ! -f boost_1_68_0.tar.gz ]; then
    clean_exec wget https://dl.bintray.com/boostorg/release/1.68.0/source/boost_1_68_0.tar.gz
  fi
  clean_exec tar vxzf boost_1_68_0.tar.gz

  pushd boost_1_68_0
  clean_exec ./bootstrap.sh --without-libraries=python --prefix=${rootdir}/3rd/boost_1_68_0
  clean_exec ./b2 -j8 cxxflags=-fPIC cflags=-fPIC -a --prefix=${rootdir}/3rd/boost_1_68_0
  clean_exec ./b2 install
  popd

  pushd ${rootdir}/3rd
  clean_exec ln -nsf boost_1_68_0 boost
  clean_exec cp ../build_tools/BUILD_boost ./boost/BUILD
  popd

  ## gflags
  if [ ! -f v2.2.1.tar.gz ]; then
    clean_exec wget https://github.com/gflags/gflags/archive/v2.2.1.tar.gz
  fi
  clean_exec tar vxzf v2.2.1.tar.gz

  pushd gflags-2.2.1
  mkdir -p build
  pushd build
  clean_exec CXXFLAGS="-fPIC" cmake .. -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=ON -DCMAKE_INSTALL_PREFIX=${rootdir}/3rd/gflags-2.2.1
  clean_exec make
  clean_exec make install
  popd
  popd

  pushd ${rootdir}/3rd
  clean_exec ln -nsf gflags-2.2.1 gflags
  clean_exec cp ../build_tools/BUILD_gflags ./gflags/BUILD
  popd

  ## glog
  if [ ! -f v0.4.0.tar.gz ]; then
    clean_exec wget https://github.com/google/glog/archive/v0.4.0.tar.gz
  fi
  clean_exec tar vxzf v0.4.0.tar.gz
  
  pushd glog-0.4.0
  clean_exec ./autogen.sh
  clean_exec CXXFLAGS='-fPIC' CFLAGS='-fPIC' ./configure --enable-shared=yes --enable-static=yes --prefix=${rootdir}/3rd/glog-0.4.0
  clean_exec GFLAGS_LIBS='' make
  clean_exec make install
  popd

  pushd ${rootdir}/3rd
  clean_exec ln -nsf glog-0.4.0 glog
  clean_exec cp ../build_tools/BUILD_glog ./glog/BUILD
  popd

  ## googletest
  if [ ! -f release-1.8.1.tar.gz ]; then
    clean_exec wget https://github.com/google/googletest/archive/release-1.8.1.tar.gz
  fi
  clean_exec tar vxzf release-1.8.1.tar.gz -C ${rootdir}/3rd 
  pushd ${rootdir}/3rd
  clean_exec ln -nsf googletest-release-1.8.1 googletest
  popd

  ## yas
  if [ ! -f 7.0.2.tar.gz ]; then
    clean_exec wget https://github.com/niXman/yas/archive/7.0.2.tar.gz
  fi
  clean_exec tar vxzf 7.0.2.tar.gz -C ${rootdir}/3rd
  pushd ${rootdir}/3rd
  clean_exec ln -nsf yas-7.0.2 yas
  clean_exec cp ../build_tools/BUILD_yas ./yas/BUILD
  popd

  ## sparsehash
  if [ ! -f sparsehash-2.0.3.tar.gz ]; then
    clean_exec wget https://github.com/sparsehash/sparsehash/archive/sparsehash-2.0.3.tar.gz
  fi
  clean_exec tar vxzf sparsehash-2.0.3.tar.gz
  pushd sparsehash-sparsehash-2.0.3
  clean_exec ./configure --prefix=${rootdir}/3rd/sparsehash-2.0.3
  clean_exec make
  clean_exec make install
  popd

  pushd ${rootdir}/3rd
  clean_exec ln -nsf sparsehash-2.0.3 sparsehash
  clean_exec cp ../build_tools/BUILD_sparsehash ./sparsehash/BUILD
  popd

  ## jni TODO

  ## hadoop TODO
  # if [ ! -f release-2.7.4.tar.gz ]; then
  #   clean_exec wget https://github.com/apache/hadoop/archive/rel/release-2.7.4.tar.gz
  # fi
  # clean_exec tar vxzf release-2.7.4.tar.gz
  # 
  # pushd hadoop-rel-release-2.7.4
  # mvn package -Pdist,native -DskipTests -Dtar
  # popd

  ## jemalloc
  if [ ! -f 5.2.0.tar.gz ]; then
    clean_exec wget https://github.com/jemalloc/jemalloc/archive/5.2.0.tar.gz
  fi
  clean_exec tar vxzf 5.2.0.tar.gz
  pushd jemalloc-5.2.0
  clean_exec ./autogen.sh
  clean_exec CXXFLAGS='-fPIC' CFLAGS='-fPIC' ./configure --enable-shared=yes --enable-static=yes --prefix=${rootdir}/3rd/jemalloc-5.2.0
  clean_exec make
  clean_exec make install
  popd

  pushd ${rootdir}/3rd
  clean_exec ln -nsf jemalloc-5.2.0 jemalloc 
  clean_exec cp ../build_tools/BUILD_jemalloc ./jemalloc/BUILD
  popd

  ## nlpack
  DLPACK_VERSION='0.2'
  if [ ! -f dlpack-${DLPACK_VERSION}.tar.gz  ]; then
    clean_exec wget https://github.com/dmlc/dlpack/archive/v${DLPACK_VERSION}.tar.gz -O dlpack-${DLPACK_VERSION}.tar.gz
  fi
  clean_exec tar vxzf dlpack-${DLPACK_VERSION}.tar.gz
  pushd dlpack-${DLPACK_VERSION}
  clean_exec mkdir -p ${rootdir}/3rd/dlpack-${DLPACK_VERSION}
  clean_exec cp include ${rootdir}/3rd/dlpack-${DLPACK_VERSION}/ -R
  popd
  pushd ${rootdir}/3rd
  clean_exec ln -nsf dlpack-${DLPACK_VERSION} dlpack
  popd

  popd
  echo "build 3rd done, you can remove .downloads now."
}

function distclean {
  pushd ${rootdir}/3rd
  clean_exec rm boost boost_1_68_0 -rf
  clean_exec rm gflags gflags-2.2.1 -rf
  clean_exec rm glog glog-0.4.0 -rf
  clean_exec rm googletest googletest-release-1.8.1 -rf
  clean_exec rm yas yas-7.0.1 -rf
  clean_exec rm jemalloc jemalloc-5.2.0 -rf 
  popd

  rm ${sourceroot} -rf

  echo "distclean 3rd done!"
}

if [ x$1 != x ]; then
  if [ $1 == "install" ]; then
    install; exit 0
  elif [ $1 == "distclean" ]; then
    distclean; exit 0
  else
    show_help; exit 1
  fi
else
  show_help; exit 1
fi

