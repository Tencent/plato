/*******************************************************************************
 *
 * Copyright (c) 2016-2018, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory
 * Written by Geoffrey M. Oxberry <oxberry1@llnl.gov>
 * LLNL-CODE-739313
 * All rights reserved.
 *
 * This file is part of gtest-mpi-listener. For details, see
 * https://github.com/LLNL/gtest-mpi-listener
 *
 * Please also see the LICENSE and COPYRIGHT files for the BSD-3
 * license and copyright information.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the disclaimer below.
 *
 * * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the disclaimer (as noted below)
 *   in the documentation and/or other materials provided with the
 *   distribution.
 *
 * * Neither the name of the LLNS/LLNL nor the names of its contributors
 *   may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL LAWRENCE
 * LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
*******************************************************************************/
//
/*******************************************************************************
 * An example from Google Test was copied with minor modifications. The
 * license of Google Test is below.
 *
 * Google Test has the following copyright notice, which must be
 * duplicated in its entirety per the terms of its license:
 *
 *  Copyright 2005, Google Inc.  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are
 *  met:
 *
 *      * Redistributions of source code must retain the above copyright
 *  notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above
 *  copyright notice, this list of conditions and the following disclaimer
 *  in the documentation and/or other materials provided with the
 *  distribution.
 *      * Neither the name of Google Inc. nor the names of its
 *  contributors may be used to endorse or promote products derived from
 *  this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*******************************************************************************/

#ifndef GTEST_MPI_MINIMAL_LISTENER_H
#define GTEST_MPI_MINIMAL_LISTENER_H

#include "mpi.h"
#include "gtest/gtest.h"
#include <cassert>
#include <vector>

// This class sets up the global test environment, which is needed
// to finalize MPI.
class MPIEnvironment : public ::testing::Environment {
public:
 MPIEnvironment() : ::testing::Environment() {}

  virtual ~MPIEnvironment() {}

  virtual void SetUp() {
    int is_mpi_initialized;
    ASSERT_EQ(MPI_Initialized(&is_mpi_initialized), MPI_SUCCESS);
    if (!is_mpi_initialized) {
      printf("MPI must be initialized before RUN_ALL_TESTS!\n");
      printf("Add '::testing::InitGoogleTest(&argc, argv);\n");
      printf("     MPI_Init(&argc, &argv);' to your 'main' function!\n");
      FAIL();
    }
  }

  virtual void TearDown() {
    int is_mpi_finalized;
    ASSERT_EQ(MPI_Finalized(&is_mpi_finalized), MPI_SUCCESS);
    if (!is_mpi_finalized) {
      int rank;
      ASSERT_EQ(MPI_Comm_rank(MPI_COMM_WORLD, &rank), MPI_SUCCESS);
      if (rank == 0) { printf("Finalizing MPI...\n"); }
      ASSERT_EQ(MPI_Finalize(), MPI_SUCCESS);
    }
    ASSERT_EQ(MPI_Finalized(&is_mpi_finalized), MPI_SUCCESS);
    ASSERT_TRUE(is_mpi_finalized);
  }

 private:
  // Disallow copying
  MPIEnvironment(const MPIEnvironment& env) {}

};

// This class more or less takes the code in Google Test's
// MinimalistPrinter example and wraps certain parts of it in MPI calls,
// gathering all results onto rank zero.
class MPIMinimalistPrinter : public ::testing::EmptyTestEventListener
{
public:
 MPIMinimalistPrinter() : ::testing::EmptyTestEventListener(),
    result_vector()
 {
    int is_mpi_initialized;
    assert(MPI_Initialized(&is_mpi_initialized) == MPI_SUCCESS);
    if (!is_mpi_initialized) {
      printf("MPI must be initialized before RUN_ALL_TESTS!\n");
      printf("Add '::testing::InitGoogleTest(&argc, argv);\n");
      printf("     MPI_Init(&argc, &argv);' to your 'main' function!\n");
      assert(0);
    }
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    UpdateCommState();
 }

 MPIMinimalistPrinter(MPI_Comm comm_) : ::testing::EmptyTestEventListener(),
    result_vector()
 {
   int is_mpi_initialized;
   assert(MPI_Initialized(&is_mpi_initialized) == MPI_SUCCESS);
   if (!is_mpi_initialized) {
     printf("MPI must be initialized before RUN_ALL_TESTS!\n");
     printf("Add '::testing::InitGoogleTest(&argc, argv);\n");
     printf("     MPI_Init(&argc, &argv);' to your 'main' function!\n");
     assert(0);
   }

   MPI_Comm_dup(comm_, &comm);
   UpdateCommState();
 }

  MPIMinimalistPrinter
    (const MPIMinimalistPrinter& printer) {

    int is_mpi_initialized;
    assert(MPI_Initialized(&is_mpi_initialized) == MPI_SUCCESS);
    if (!is_mpi_initialized) {
      printf("MPI must be initialized before RUN_ALL_TESTS!\n");
      printf("Add '::testing::InitGoogleTest(&argc, argv);\n");
      printf("     MPI_Init(&argc, &argv);' to your 'main' function!\n");
      assert(0);
    }

    MPI_Comm_dup(printer.comm, &comm);
    UpdateCommState();
    result_vector = printer.result_vector;
  }

  ~MPIMinimalistPrinter() {
    int is_mpi_finalized;
    assert(MPI_Finalized(&is_mpi_finalized) == MPI_SUCCESS);
    if (!is_mpi_finalized) {
      MPI_Comm_free(&comm);
    }
  }

  // Called before a test starts.
  virtual void OnTestStart(const ::testing::TestInfo& test_info) {
    // Only need to report test start info on rank 0
    if (rank == 0) {
      printf("*** Test %s.%s starting.\n",
             test_info.test_case_name(), test_info.name());
    }
  }

  // Called after an assertion failure or an explicit SUCCESS() macro.
  // In an MPI program, this means that certain ranks may not call this
  // function if a test part does not fail on all ranks. Consequently, it
  // is difficult to have explicit synchronization points here.
  virtual void OnTestPartResult
    (const ::testing::TestPartResult& test_part_result) {
    result_vector.push_back(test_part_result);
  }

  // Called after a test ends.
  virtual void OnTestEnd(const ::testing::TestInfo& test_info) {
    int localResultCount = result_vector.size();
    std::vector<int> resultCountOnRank(size, 0);
    MPI_Gather(&localResultCount, 1, MPI_INT,
               &resultCountOnRank[0], 1, MPI_INT,
               0, comm);

    if (rank != 0) {
      // Nonzero ranks send constituent parts of each result to rank 0
      for (int i = 0; i < localResultCount; i++) {
        const ::testing::TestPartResult test_part_result = result_vector.at(i);
        int resultStatus(test_part_result.failed());
        std::string resultFileName(test_part_result.file_name());
        int resultLineNumber(test_part_result.line_number());
        std::string resultSummary(test_part_result.summary());

        // Must add one for null termination
        int resultFileNameSize(resultFileName.size()+1);
        int resultSummarySize(resultSummary.size()+1);

        MPI_Send(&resultStatus, 1, MPI_INT, 0, rank, comm);
        MPI_Send(&resultFileNameSize, 1, MPI_INT, 0, rank, comm);
        MPI_Send(&resultLineNumber, 1, MPI_INT, 0, rank, comm);
        MPI_Send(&resultSummarySize, 1, MPI_INT, 0, rank, comm);
        MPI_Send((char *)resultFileName.c_str(), resultFileNameSize, MPI_CHAR,
                 0, rank, comm);
        MPI_Send((char *)resultSummary.c_str(), resultSummarySize, MPI_CHAR,
                 0, rank, comm);
      }
    } else {
      // Rank 0 first prints its local result data
      for (int i = 0; i < localResultCount; i++) {
        const ::testing::TestPartResult test_part_result = result_vector.at(i);
        printf("      %s on rank %d, %s:%d\n%s\n",
             test_part_result.failed() ? "*** Failure" : "Success",
             rank,
             test_part_result.file_name(),
             test_part_result.line_number(),
             test_part_result.summary());
      }

      for (int r = 1; r < size; r++) {
        for (int i = 0; i < resultCountOnRank[r]; i++) {
          int resultStatus, resultFileNameSize, resultLineNumber, resultSummarySize;
          MPI_Recv(&resultStatus, 1, MPI_INT, r, r, comm, MPI_STATUS_IGNORE);
          MPI_Recv(&resultFileNameSize, 1, MPI_INT, r, r, comm,
                   MPI_STATUS_IGNORE);
          MPI_Recv(&resultLineNumber, 1, MPI_INT, r, r, comm,
                   MPI_STATUS_IGNORE);
          MPI_Recv(&resultSummarySize, 1, MPI_INT, r, r, comm,
                   MPI_STATUS_IGNORE);
          char resultFileName[resultFileNameSize];
          char resultSummary[resultSummarySize];
          MPI_Recv(resultFileName, resultFileNameSize, MPI_CHAR, r, r, comm,
                   MPI_STATUS_IGNORE);
          MPI_Recv(resultSummary, resultSummarySize, MPI_CHAR, r, r, comm,
                   MPI_STATUS_IGNORE);

          printf("      %s on rank %d, %s:%d\n%s\n",
                 resultStatus ? "*** Failure" : "Success",
                 r,
                 resultFileName,
                 resultLineNumber,
                 resultSummary);
        }
      }

      printf("*** Test %s.%s ending.\n",
             test_info.test_case_name(), test_info.name());
    }

    result_vector.clear();
}

 private:
  MPI_Comm comm;
  int rank;
  int size;
  std::vector< ::testing::TestPartResult > result_vector;

  int UpdateCommState()
  {
    int flag = MPI_Comm_rank(comm, &rank);
    if (flag != MPI_SUCCESS) { return flag; }
    flag = MPI_Comm_size(comm, &size);
    return flag;
  }

};

#endif /* GTEST_MPI_MINIMAL_LISTENER_H */
