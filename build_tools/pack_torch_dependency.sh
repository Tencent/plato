#!/bin/bash

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/../"
PACKAGE=${ROOT}/torch_dependency.tar
MKL_LIB_PATH="${ROOT}/3rd/mkl/mkl/lib/intel64"
INTEL_LIB_PATH="${ROOT}/3rd/mkl/lib/intel64"

TORCH_SO='libc10.so libcaffe2.so libcaffe2_detectron_ops.so libcaffe2_observers.so libfoxi.so libonnxifi.so libshm.so libtorch.so libtorch.so.1'
MKL_SO='libmkl_def.so libmkl_avx2.so libmkl_intel_lp64.so libmkl_intel_thread.so libmkl_core.so libmkl_gnu_thread.so'
MKL_SO="${MKL_SO} libmkl_vml_def.so libmkl_vml_avx2.so   libmkl_vml_mc3.so libmkl_vml_cmpt.so"
INTEL_SO='libiomp5.so'

echo "pack everything into ${PACKAGE}"

pushd ${ROOT}

mkdir -p lib

# cp libtorch to .lib
pushd ${ROOT}/3rd/libtorch/lib/
cp ${TORCH_SO} ${ROOT}/lib
popd

# pack intel
pushd ${INTEL_LIB_PATH}
cp ${INTEL_SO} ${ROOT}/lib
popd

# pack mkl
pushd ${MKL_LIB_PATH}
cp ${MKL_SO} ${ROOT}/lib
popd

tar -vcf ${PACKAGE} lib

# compress at the end
gzip ${PACKAGE}

rm -rf lib

popd

