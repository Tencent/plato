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

#include <iostream>
#include <string.h>
#include <immintrin.h>

/**
 * @brief gen mask_8x32
 */
void gen_shuffle_mask_8x32() {
  for (unsigned i = 0; i < 256; ++i) {
    int count = 0;
    int permutation[8];
    memset(permutation, 0xFF, sizeof(permutation));
    for (unsigned b = 0; b < 8; ++b) {
      if (i & (1 << b)) {
        permutation[count++] = b;
      }
    }

    std::cout << "(__m256i)(__v8si){";

    for (unsigned j = 0; j < 8; j++) {
      if (j == 7) {
        std::cout << permutation[j];
      } else {
        std::cout << permutation[j] << ",";
      }
    }
    std::cout << "}," << std::endl;
  }
}

/**
 * @brief gen mask_8x16
 */
void gen_shuffle_mask_8x16() {
  for(unsigned i = 0; i < 256; i++) {
    int counter = 0;
    char permutation[16];
    memset(permutation, 0xFF, sizeof(permutation));
    for(char b = 0; b < 8; b++) {
      if(i & (1 << b)) {
        permutation[counter++] = 2*b;
        permutation[counter++] = 2*b + 1;
      }
    }

    std::cout << "(__m128i)(__v16qi){";
    for (int j = 0; j < 16; j++) {
      if (j == 15) {
        std::cout << (int)permutation[j];
      } else {
        std::cout << (int)permutation[j] << ",";
      }
    }
    std::cout << "}," << std::endl;
  }
}

/**
 * @brief gen mask_2x64
 */
void gen_shuffle_mask_2x64() {
  for(unsigned i = 0; i < 4; i++) {
    int counter = 0;
    char permutation[16];
    memset(permutation, 0xFF, sizeof(permutation));
    for(char b = 0; b < 4; b++) {
      if(i & (1 << b)) {
        permutation[counter++] = 8*b;
        permutation[counter++] = 8*b + 1;
        permutation[counter++] = 8*b + 2;
        permutation[counter++] = 8*b + 3;
        permutation[counter++] = 8*b + 4;
        permutation[counter++] = 8*b + 5;
        permutation[counter++] = 8*b + 6;
        permutation[counter++] = 8*b + 7;
      }
    }

    std::cout << "(__m128i)(__v16qi){";
    for (int j = 0; j < 16; j++) {
      if (j == 15) {
        std::cout << (int)permutation[j];
      } else {
        std::cout << (int)permutation[j] << ",";
      }
    }
    std::cout << "}," << std::endl;
  }
}

/**
 * @brief gen mask_4x32
 */
void gen_shuffle_mask_4x32() {
  for(unsigned i = 0; i < 16; i++) {
    int counter = 0;
    char permutation[16];
    memset(permutation, 0xFF, sizeof(permutation));
    for(char b = 0; b < 4; b++) {
      if(i & (1 << b)) {
        permutation[counter++] = 4*b;
        permutation[counter++] = 4*b + 1;
        permutation[counter++] = 4*b + 2;
        permutation[counter++] = 4*b + 3;
      }
    }

    std::cout << "(__m128i)(__v16qi){";
    for (int j = 0; j < 16; j++) {
      if (j == 15) {
        std::cout << (int)permutation[j];
      } else {
        std::cout << (int)permutation[j] << ",";
      }
    }
    std::cout << "}," << std::endl;
  }
}

/**
 * @brief gen mask_4x64
 */
void gen_shuffle_mask_4x64() {
  for(unsigned i = 0; i < 16; i++) {
    int counter = 0;
    int32_t permutation[8];
    memset(permutation, 0xFF, sizeof(permutation));
    for(char b = 0; b < 4; b++) {
      if(i & (1 << b)) {
        permutation[counter++] = 2*b;
        permutation[counter++] = 2*b + 1;
      }
    }

    std::cout << "(__m256i)(__v8si){";
    for (int j = 0; j < 8; j++) {
      if (j == 7) {
        std::cout << (int)permutation[j];
      } else {
        std::cout << (int)permutation[j] << ",";
      }
    }
    std::cout << "}," << std::endl;
  }
}

int main() {
  gen_shuffle_mask_8x16();
  gen_shuffle_mask_8x32();
  gen_shuffle_mask_4x32();
  gen_shuffle_mask_4x64();
  gen_shuffle_mask_2x64();
  return 0;
}