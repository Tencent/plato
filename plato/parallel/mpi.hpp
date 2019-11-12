/*
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

// class mpi_instance_t {
// public:
//   int partition_id_;
//   int partitions_;
// 
//   mpi_instance_t (int* argc, char*** argv) {
//     int provided;
//     MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided);
// 
//     MPI_Comm_rank(MPI_COMM_WORLD, &partition_id_);
//     MPI_Comm_size(MPI_COMM_WORLD, &partitions_);
// 
//     if (0 == partition_id_) {
//       LOG(INFO) << "thread support level provided by MPI: ";
//       switch (provided) {
//       case MPI_THREAD_MULTIPLE:
//         LOG(INFO) << "MPI_THREAD_MULTIPLE";
//         break;
//       case MPI_THREAD_SERIALIZED:
//         LOG(INFO) << "MPI_THREAD_SERIALIZED";
//         break;
//       case MPI_THREAD_FUNNELED:
//         LOG(INFO) << "MPI_THREAD_FUNNELED";
//         break;
//       case MPI_THREAD_SINGLE:
//         LOG(INFO) << "MPI_THREAD_SINGLE";
//         break;
//       default:
//         CHECK(false);
//       }
//     }
//   }
// 
//   ~mpi_instance_t() { MPI_Finalize(); }
// };

}

#endif

