/*
Copyright (c) 2015-2016 Xiaowei Zhu, Tsinghua University

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
   

  Tencent is pleased to support the open source community by making
  Plato available.
  Copyright (C) 2019 THL A29 Limited, a Tencent company.
  All rights reserved.

  Licensed under the BSD 3-Clause License (the "License"); you may
  not use this file except in compliance with the License. You may
  obtain a copy of the License at

  https://opensource.org/licenses/BSD-3-Clause

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" basis,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
  implied. See the License for the specific language governing
  permissions and limitations under the License.

  See the AUTHORS file for names of contributors.
*/

#ifndef __PLATO_MPI_HPP__
#define __PLATO_MPI_HPP__

#include <cstdint>
#include <cstdlib>

#include <type_traits>

#include "mpi.h"
#include "glog/logging.h"

#define MPI_MSG_MAX_SIZE 1073741824 

namespace plato {

enum MessageTag {
  Stop        = 0,
  PassMessage = 1,
  Shuffle     = 2,
  ShuffleFin  = 3
};

template <typename T>
MPI_Datatype get_mpi_data_type() {
  if (std::is_same<T, char>::value) {
    return MPI_CHAR;
  } else if (std::is_same<T, unsigned char>::value) {
    return MPI_UNSIGNED_CHAR;
  } else if (std::is_same<T, int>::value) {
    return MPI_INT;
  } else if (std::is_same<T, unsigned>::value) {
    return MPI_UNSIGNED;
  } else if (std::is_same<T, int16_t>::value) {
    return MPI_INT16_T;
  } else if (std::is_same<T, uint16_t>::value) {
    return MPI_UINT16_T;
  } else if (std::is_same<T, long>::value) {
    return MPI_LONG;
  } else if (std::is_same<T, unsigned long>::value) {
    return MPI_UNSIGNED_LONG;
  } else if (std::is_same<T, float>::value) {
    return MPI_FLOAT;
  } else if (std::is_same<T, double>::value) {
    return MPI_DOUBLE;
  } else {
    CHECK(false) << "type not supported";
  }
}

int bcast(void *buffer, size_t count, MPI_Datatype datatype, int root,
    MPI_Comm comm) {  
  //mpi_bcast will fail when message size large than 2GB
  int type_size;
  MPI_Type_size(datatype, &type_size);
  size_t message_size = count * (size_t)type_size;
  if (message_size >= MPI_MSG_MAX_SIZE) {
    size_t max_count = MPI_MSG_MAX_SIZE / type_size;
    for (size_t i = 0; i < count; i += max_count) {
      size_t actual_count = max_count;
      if (i + actual_count > count) {
        actual_count = count - i;
      }
      auto ptr = (char*)buffer + i * (size_t)type_size;
      int rc = MPI_Bcast(ptr, actual_count, datatype, root, comm);
      if (rc != MPI_SUCCESS) return rc;
    }
  } else {
    return MPI_Bcast(buffer, count, datatype, root, comm);
  }

  return MPI_SUCCESS;
}

int allreduce(const void *send_buf, void *recv_buf, size_t count,
    MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
  //mpi_allreduce will fail when message size is too large
  int type_size;
  MPI_Type_size(datatype, &type_size);
  size_t message_size = count * (size_t)type_size;
  if (message_size >= MPI_MSG_MAX_SIZE) {
    size_t max_count = MPI_MSG_MAX_SIZE / type_size;
    for (size_t i = 0; i < count; i += max_count) {
      size_t actual_count = max_count;
      if (i + actual_count > count) {
        actual_count = count - i;
      }
      auto ptr = send_buf;
      if (ptr != MPI_IN_PLACE) {
        ptr = (const char*)send_buf + i * (size_t)type_size;
      }
      int rc = MPI_Allreduce(ptr, (char*)recv_buf + i * (size_t)type_size, 
          actual_count, datatype, op, comm);
      if (rc != MPI_SUCCESS) return rc;
    }
  } else {
    return MPI_Allreduce(send_buf, recv_buf, count, datatype, op, comm);
  }

  return MPI_SUCCESS;
}

}

#endif

