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

/**
 * references:
 *
 * 1.https://highlyscalable.wordpress.com/2012/06/05/fast-intersection-sorted-lists-sse/
 * 2.Schlegel B, Willhalm T, Lehner W. Fast Sorted-Set Intersection using SIMD Instructions[C]//ADMS@ VLDB. 2011: 1-8.
 * 3.Lemire D, Boytsov L, Kurz N. SIMD compression and the intersection of sorted integers[J]. Software: Practice and Experience, 2016, 46(6): 723-749.
 * 4.Inoue H, Ohara M, Taura K. Faster set intersection with SIMD instructions by reducing branch mispredictions[J]. Proceedings of the VLDB Endowment, 2014, 8(3): 293-304.
 * 5.Han S, Zou L, Yu J X. Speeding Up Set Intersections in Graph Algorithms using SIMD Instructions[C]//Proceedings of the 2018 International Conference on Management of Data. ACM, 2018: 1587-1602.
 *
 */

#pragma once

#ifdef __AVX512F__
#ifndef __USE_AVX512__
#define __USE_AVX512__
#endif
#elif defined(__USE_AVX512__)
#pragma GCC push_options
#pragma GCC target("avx512f")
#define __DIASBLE_USE_AVX512__
#endif

#pragma GCC optimize("O3")

#include <unistd.h>

namespace plato {

// Scalar:

/**
 * @brief intersection of two set. should be sorted.
 * @tparam T - element type: must be number type.
 * @tparam SIZE_T - size type: must be number type.
 * @param set_a - pointer to one sorted set.
 * @param size_a - size of one sorted set.
 * @param set_b - pointer to another sorted set.
 * @param size_b - size of another sorted set.
 * @param out - output pointer.
 * @return result size.
 */
template <typename T, typename SIZE_T>
inline
SIZE_T intersect_scalar(const T *set_a, SIZE_T size_a, const T *set_b, SIZE_T size_b, T *out);

/**
 * @brief intersection of two set. should be sorted.
 * @tparam T - element type: must be number type.
 * @tparam SIZE_T - size type: must be number type.
 * @param set_a - pointer to one sorted set.
 * @param size_a - size of one sorted set.
 * @param set_b - pointer to another sorted set.
 * @param size_b - size of another sorted set.
 * @param out - output pointer.
 * @return result size.
 */
template <typename T, typename SIZE_T, typename = typename std::enable_if<sizeof(T) == 4 || sizeof(T) == 8, std::true_type>::type>
inline
SIZE_T intersect_simd_galloping(const T *set_a, SIZE_T size_a, const T *set_b, SIZE_T size_b, T *out);

/**
 * @brief intersection of two set. should be sorted.
 * @tparam T - element type: must be number type.
 * @tparam SIZE_T - size type: must be number type.
 * @param set_a - pointer to one sorted set.
 * @param size_a - size of one sorted set.
 * @param set_b - pointer to another sorted set.
 * @param size_b - size of another sorted set.
 * @param out - output pointer.
 * @return result size.
 */
// Shuffling:
template <typename T, typename SIZE_T, typename = typename std::enable_if<sizeof(T) == 4 || sizeof(T) == 8, std::true_type>::type>
inline
SIZE_T intersect_simd_shuffle(const T *set_a, SIZE_T size_a, const T *set_b, SIZE_T size_b, T *out);

/**
 * @brief intersection of two set. should be sorted.
 * @tparam T - element type: must be number type.
 * @tparam SIZE_T - size type: must be number type.
 * @param set_a - pointer to one sorted set.
 * @param size_a - size of one sorted set.
 * @param set_b - pointer to another sorted set.
 * @param size_b - size of another sorted set.
 * @param out - output pointer.
 * @return result size.
 */
template <typename T, typename SIZE_T, typename = typename std::enable_if<sizeof(T) == 4 || sizeof(T) == 8, std::true_type>::type>
inline
SIZE_T intersect_simd_shuffle_x2(const T *set_a, SIZE_T size_a, const T *set_b, SIZE_T size_b, T *out);

/**
 * @brief intersection of two set. should be sorted.
 * @tparam T - element type: must be number type.
 * @tparam SIZE_T - size type: must be number type.
 * @param set_a - pointer to one sorted set.
 * @param size_a - size of one sorted set.
 * @param set_b - pointer to another sorted set.
 * @param size_b - size of another sorted set.
 * @param out - output pointer.
 * @return result size.
 */
template <typename T, typename SIZE_T, typename = typename std::enable_if<sizeof(T) == 2, std::true_type>::type>
inline
SIZE_T intersect_simd_sttni(const T *set_a, SIZE_T size_a, const T *set_b, SIZE_T size_b, T *out);

/**
 * @brief intersection of two set. should be sorted.
 * @tparam T - element type: must be number type.
 * @tparam SIZE_T - size type: must be number type.
 * @param set_a - pointer to one sorted set.
 * @param size_a - size of one sorted set.
 * @param set_b - pointer to another sorted set.
 * @param size_b - size of another sorted set.
 * @param out - output pointer.
 * @return result size.
 */
template <typename T, typename SIZE_T, typename = typename std::enable_if<sizeof(T) == 2, std::true_type>::type>
inline
SIZE_T intersect_simd_sttni_x2(const T *set_a, SIZE_T size_a, const T *set_b, SIZE_T size_b, T *out);

/**
 * @brief intersection of two set. should be sorted.
 * @tparam T - element type: must be number type.
 * @tparam SIZE_T - size type: must be number type.
 * @param set_a - pointer to one sorted set.
 * @param size_a - size of one sorted set.
 * @param set_b - pointer to another sorted set.
 * @param size_b - size of another sorted set.
 * @param out - output pointer.
 * @return result size.
 */
template <typename T, typename SIZE_T, typename = typename std::enable_if<sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8, std::true_type>::type>
inline
SIZE_T intersect_simd(const T *set_a, SIZE_T size_a, const T *set_b, SIZE_T size_b, T *out);

#ifdef __AVX2__

/**
 * @brief intersection of two set. should be sorted.
 * @tparam T - element type: must be number type.
 * @tparam SIZE_T - size type: must be number type.
 * @param set_a - pointer to one sorted set.
 * @param size_a - size of one sorted set.
 * @param set_b - pointer to another sorted set.
 * @param size_b - size of another sorted set.
 * @param out - output pointer.
 * @return result size.
 */
template <typename T, typename SIZE_T, typename = typename std::enable_if<sizeof(T) == 4 || sizeof(T) == 8, std::true_type>::type>
inline
SIZE_T intersect_simd_shuffle_avx(const T *set_a, SIZE_T size_a, const T *set_b, SIZE_T size_b, T *out);

/**
 * @brief intersection of two set. should be sorted.
 * @tparam T - element type: must be number type.
 * @tparam SIZE_T - size type: must be number type.
 * @param set_a - pointer to one sorted set.
 * @param size_a - size of one sorted set.
 * @param set_b - pointer to another sorted set.
 * @param size_b - size of another sorted set.
 * @param out - output pointer.
 * @return result size.
 */
template <typename T, typename SIZE_T, typename = typename std::enable_if<sizeof(T) == 4 || sizeof(T) == 8, std::true_type>::type>
inline
SIZE_T intersect_simd_shuffle_avx_x2(const T *set_a, SIZE_T size_a, const T *set_b, SIZE_T size_b, T *out);

/**
 * @brief intersection of two set. should be sorted.
 * @tparam T - element type: must be number type.
 * @tparam SIZE_T - size type: must be number type.
 * @param set_a - pointer to one sorted set.
 * @param size_a - size of one sorted set.
 * @param set_b - pointer to another sorted set.
 * @param size_b - size of another sorted set.
 * @param out - output pointer.
 * @return result size.
 */
template <typename T, typename SIZE_T, typename = typename std::enable_if<sizeof(T) == 4 || sizeof(T) == 8, std::true_type>::type>
inline
SIZE_T intersect_simd_galloping_avx(const T *set_a, SIZE_T size_a, const T *set_b, SIZE_T size_b, T *out);

/**
 * @brief intersection of two set. should be sorted.
 * @tparam T - element type: must be number type.
 * @tparam SIZE_T - size type: must be number type.
 * @param set_a - pointer to one sorted set.
 * @param size_a - size of one sorted set.
 * @param set_b - pointer to another sorted set.
 * @param size_b - size of another sorted set.
 * @param out - output pointer.
 * @return result size.
 */
template <typename T, typename SIZE_T, typename = typename std::enable_if<sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8, std::true_type>::type>
inline
SIZE_T intersect_simd_avx(const T *set_a, SIZE_T size_a, const T *set_b, SIZE_T size_b, T *out);

#endif

#ifdef __USE_AVX512__

/**
 * @brief intersection of two set. should be sorted.
 * @tparam T - element type: must be number type.
 * @tparam SIZE_T - size type: must be number type.
 * @param set_a - pointer to one sorted set.
 * @param size_a - size of one sorted set.
 * @param set_b - pointer to another sorted set.
 * @param size_b - size of another sorted set.
 * @param out - output pointer.
 * @return result size.
 */
template <typename T, typename SIZE_T, typename = typename std::enable_if<sizeof(T) == 4 || sizeof(T) == 8, std::true_type>::type>
inline
SIZE_T intersect_simd_shuffle_avx512(const T *set_a, SIZE_T size_a, const T *set_b, SIZE_T size_b, T *out);

/**
 * @brief intersection of two set. should be sorted.
 * @tparam T - element type: must be number type.
 * @tparam SIZE_T - size type: must be number type.
 * @param set_a - pointer to one sorted set.
 * @param size_a - size of one sorted set.
 * @param set_b - pointer to another sorted set.
 * @param size_b - size of another sorted set.
 * @param out - output pointer.
 * @return result size.
 */
template <typename T, typename SIZE_T, typename = typename std::enable_if<sizeof(T) == 4 || sizeof(T) == 8, std::true_type>::type>
inline
SIZE_T intersect_simd_galloping_avx512(const T *set_a, SIZE_T size_a, const T *set_b, SIZE_T size_b, T *out);

/**
 * @brief intersection of two set. should be sorted.
 * @tparam T - element type: must be number type.
 * @tparam SIZE_T - size type: must be number type.
 * @param set_a - pointer to one sorted set.
 * @param size_a - size of one sorted set.
 * @param set_b - pointer to another sorted set.
 * @param size_b - size of another sorted set.
 * @param out - output pointer.
 * @return result size.
 */
template <typename T, typename SIZE_T, typename = typename std::enable_if<sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8, std::true_type>::type>
inline
SIZE_T intersect_simd_avx512(const T *set_a, SIZE_T size_a, const T *set_b, SIZE_T size_b, T *out);

#endif

/**
 * @brief intersection of two set. should be sorted.
 * @tparam T - element type: must be number type.
 * @tparam SIZE_T - size type: must be number type.
 * @param set_a - pointer to one sorted set.
 * @param size_a - size of one sorted set.
 * @param set_b - pointer to another sorted set.
 * @param size_b - size of another sorted set.
 * @param out - output pointer.
 * @return result size.
 */
template <typename T, typename SIZE_T, typename = typename std::enable_if<sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8, std::true_type>::type>
inline
SIZE_T intersect(const T *set_a, SIZE_T size_a, const T *set_b, SIZE_T size_b, T *out);

}

#include "intersection_impl.hpp"

#ifdef __DIASBLE_USE_AVX512__
#pragma GCC pop_options
#endif