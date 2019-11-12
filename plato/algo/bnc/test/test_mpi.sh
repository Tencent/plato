#!/bin/bash

# Copyright (c) 2018-2019 Wechat Group, Tencent
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e
export OMP_NUM_THREADS=2

PROG="
$PLATO_ROOT/bazel-bin/plato/algo/bnc/test/test_bnc
"
echo $PROG
if [[ ! -z $PLATO_ROOT ]]; then
  cd $PLATO_ROOT
  for BIN in $PROG; do
    CMD="mpirun -np 4 $BIN "
    for arg in $*
    do
      CMD=$CMD" "$arg
    done
    echo $CMD
    eval $CMD
  done
fi
