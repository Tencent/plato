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

#include <immintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <unistd.h>

namespace plato {

extern const __m128i shuffle_mask_4x32[16];
extern const __m128i shuffle_mask_8x16[256];
extern const __m128i shuffle_mask_2x64[4];
extern const __m256i shuffle_mask_8x32[256];
extern const __m256i shuffle_mask_4x64[16];
extern const bool cpu_support_avx512;

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
SIZE_T intersect_scalar(const T *set_a, const SIZE_T size_a, const T *set_b, const SIZE_T size_b, T *out) {
  SIZE_T i = 0, j = 0, size_out = 0;
  while (i < size_a && j < size_b) {
    if (set_a[i] == set_b[j]) {
      out[size_out++] = set_a[i];
      i++; j++;
    } else if (set_a[i] < set_b[j]) {
      i++;
    } else {
      j++;
    }
  }
  return size_out;
}


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
template <typename T, typename SIZE_T, typename>
SIZE_T intersect_simd_galloping(const T *set_a, const SIZE_T size_a, const T *set_b, const SIZE_T size_b, T *out) {
  constexpr static size_t veclen = sizeof(__m128i) / sizeof(T);
  const T *set_freq = size_a < size_b ? set_b : set_a;
  const SIZE_T size_freq = size_a < size_b ? size_b : size_a;
  const T *set_rare = size_a < size_b ? set_a : set_b;
  const SIZE_T size_rare = size_a < size_b ? size_a : size_b;

  const SIZE_T qs_freq = size_freq & ~(veclen * 32 - 1);
  const SIZE_T qs_rare = size_rare;
  SIZE_T i = 0, j = 0, size_out = 0;

  while (i < qs_freq && j < qs_rare) {
    const T match_rare = set_rare[j];
    if (set_freq[i + veclen * 32 - 1] < match_rare) {
      i += veclen * 32;
      continue;
    }

    __m128i test;

    if (sizeof(T) == 4) {
      const T match_rare_tmp = match_rare;
      const __m128i match = _mm_set1_epi32(match_rare_tmp);

      if (set_freq[i + veclen * 16 - 1] >= match_rare) {
        if (set_freq[i + veclen * 8 - 1] < match_rare) {
          test = _mm_or_si128(
            _mm_or_si128(
              _mm_or_si128(
                _mm_cmpeq_epi32(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 8), match),
                _mm_cmpeq_epi32(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 9), match)
              ),
              _mm_or_si128(
                _mm_cmpeq_epi32(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 10), match),
                _mm_cmpeq_epi32(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 11), match)
              )),
            _mm_or_si128(
              _mm_or_si128(
                _mm_cmpeq_epi32(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 12), match),
                _mm_cmpeq_epi32(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 13), match)
              ),
              _mm_or_si128(
                _mm_cmpeq_epi32(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 14), match),
                _mm_cmpeq_epi32(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 15), match)
              )
            )
          );
        } else {
          test = _mm_or_si128(
            _mm_or_si128(
              _mm_or_si128(
                _mm_cmpeq_epi32(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 0), match),
                _mm_cmpeq_epi32(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 1), match)
              ),
              _mm_or_si128(
                _mm_cmpeq_epi32(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 2), match),
                _mm_cmpeq_epi32(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 3), match)
              )),
            _mm_or_si128(
              _mm_or_si128(
                _mm_cmpeq_epi32(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 4), match),
                _mm_cmpeq_epi32(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 5), match)
              ),
              _mm_or_si128(
                _mm_cmpeq_epi32(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 6), match),
                _mm_cmpeq_epi32(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 7), match)
              )
            )
          );
        }
      } else {
        if (set_freq[i + veclen * 24 - 1] < match_rare) {
          test = _mm_or_si128(
            _mm_or_si128(
              _mm_or_si128(
                _mm_cmpeq_epi32(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 8 + 16), match),
                _mm_cmpeq_epi32(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 9 + 16), match)
              ),
              _mm_or_si128(
                _mm_cmpeq_epi32(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 10 + 16), match),
                _mm_cmpeq_epi32(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 11 + 16), match)
              )),
            _mm_or_si128(
              _mm_or_si128(
                _mm_cmpeq_epi32(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 12 + 16), match),
                _mm_cmpeq_epi32(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 13 + 16), match)
              ),
              _mm_or_si128(
                _mm_cmpeq_epi32(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 14 + 16), match),
                _mm_cmpeq_epi32(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 15 + 16), match)
              )
            )
          );
        } else {
          test = _mm_or_si128(
            _mm_or_si128(
              _mm_or_si128(
                _mm_cmpeq_epi32(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 0 + 16), match),
                _mm_cmpeq_epi32(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 1 + 16), match)
              ),
              _mm_or_si128(
                _mm_cmpeq_epi32(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 2 + 16), match),
                _mm_cmpeq_epi32(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 3 + 16), match)
              )),
            _mm_or_si128(
              _mm_or_si128(
                _mm_cmpeq_epi32(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 4 + 16), match),
                _mm_cmpeq_epi32(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 5 + 16), match)
              ),
              _mm_or_si128(
                _mm_cmpeq_epi32(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 6 + 16), match),
                _mm_cmpeq_epi32(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 7 + 16), match)
              )
            )
          );
        }
      }
    } else if (sizeof(T) == 8) {
      const long long match_rare_tmp = match_rare;
      const __m128i match = _mm_set1_epi64((__m64)match_rare_tmp);

      if (set_freq[i + veclen * 16 - 1] >= match_rare) {
        if (set_freq[i + veclen * 8 - 1] < match_rare) {
          test = _mm_or_si128(
            _mm_or_si128(
              _mm_or_si128(
                _mm_cmpeq_epi64(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 8), match),
                _mm_cmpeq_epi64(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 9), match)
              ),
              _mm_or_si128(
                _mm_cmpeq_epi64(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 10), match),
                _mm_cmpeq_epi64(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 11), match)
              )),
            _mm_or_si128(
              _mm_or_si128(
                _mm_cmpeq_epi64(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 12), match),
                _mm_cmpeq_epi64(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 13), match)
              ),
              _mm_or_si128(
                _mm_cmpeq_epi64(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 14), match),
                _mm_cmpeq_epi64(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 15), match)
              )
            )
          );
        } else {
          test = _mm_or_si128(
            _mm_or_si128(
              _mm_or_si128(
                _mm_cmpeq_epi64(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 0), match),
                _mm_cmpeq_epi64(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 1), match)
              ),
              _mm_or_si128(
                _mm_cmpeq_epi64(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 2), match),
                _mm_cmpeq_epi64(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 3), match)
              )),
            _mm_or_si128(
              _mm_or_si128(
                _mm_cmpeq_epi64(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 4), match),
                _mm_cmpeq_epi64(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 5), match)
              ),
              _mm_or_si128(
                _mm_cmpeq_epi64(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 6), match),
                _mm_cmpeq_epi64(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 7), match)
              )
            )
          );
        }
      } else {
        if (set_freq[i + veclen * 24 - 1] < match_rare) {
          test = _mm_or_si128(
            _mm_or_si128(
              _mm_or_si128(
                _mm_cmpeq_epi64(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 8 + 16), match),
                _mm_cmpeq_epi64(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 9 + 16), match)
              ),
              _mm_or_si128(
                _mm_cmpeq_epi64(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 10 + 16), match),
                _mm_cmpeq_epi64(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 11 + 16), match)
              )),
            _mm_or_si128(
              _mm_or_si128(
                _mm_cmpeq_epi64(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 12 + 16), match),
                _mm_cmpeq_epi64(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 13 + 16), match)
              ),
              _mm_or_si128(
                _mm_cmpeq_epi64(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 14 + 16), match),
                _mm_cmpeq_epi64(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 15 + 16), match)
              )
            )
          );
        } else {
          test = _mm_or_si128(
            _mm_or_si128(
              _mm_or_si128(
                _mm_cmpeq_epi64(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 0 + 16), match),
                _mm_cmpeq_epi64(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 1 + 16), match)
              ),
              _mm_or_si128(
                _mm_cmpeq_epi64(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 2 + 16), match),
                _mm_cmpeq_epi64(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 3 + 16), match)
              )),
            _mm_or_si128(
              _mm_or_si128(
                _mm_cmpeq_epi64(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 4 + 16), match),
                _mm_cmpeq_epi64(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 5 + 16), match)
              ),
              _mm_or_si128(
                _mm_cmpeq_epi64(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 6 + 16), match),
                _mm_cmpeq_epi64(_mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_freq + i) + 7 + 16), match)
              )
            )
          );
        }
      }
    } else {
      abort();
    }

#ifdef __SSE4_1__
    if (!_mm_testz_si128(test, test))
#else
      if (movemask_epi8(test))
#endif
    {
      out[size_out++] = match_rare;
    }

    ++j;
  }

  return size_out + intersect_simd_shuffle(set_freq + i, size_freq - i, set_rare + j, size_rare - j, out + size_out);
}

// Shuffling:
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
template <typename T, typename SIZE_T, typename>
SIZE_T intersect_simd_shuffle(const T *set_a, const SIZE_T size_a, const T *set_b, const SIZE_T size_b, T *out) {
  constexpr static size_t veclen = sizeof(__m128i) / sizeof(T);

  const SIZE_T qs_a = size_a - (size_a & (veclen - 1));
  const SIZE_T qs_b = size_b - (size_b & (veclen - 1));
  SIZE_T i = 0, j = 0, size_out = 0;

  while (i < qs_a && j < qs_b) {
    const __m128i v_a = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_a + i));
    const __m128i v_b = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_b + j));

    const T a_max = set_a[i + veclen - 1];
    const T b_max = set_b[j + veclen - 1];
    if (a_max == b_max) {
      i += veclen;
      j += veclen;
      _mm_prefetch(set_a + i, _MM_HINT_NTA);
      _mm_prefetch(set_b + j, _MM_HINT_NTA);
    } else if (a_max < b_max) {
      i += veclen;
      _mm_prefetch(set_a + i, _MM_HINT_NTA);
    } else {
      j += veclen;
      _mm_prefetch(set_b + j, _MM_HINT_NTA);
    }

    if (sizeof(T) == 4) {
      int mask = _mm_movemask_ps((__m128)_mm_or_si128(
        _mm_or_si128(
          _mm_cmpeq_epi32(v_a, v_b),
          _mm_cmpeq_epi32(v_a, _mm_shuffle_epi32(v_b, _MM_SHUFFLE(0,3,2,1)))
        ),
        _mm_or_si128(
          _mm_cmpeq_epi32(v_a, _mm_shuffle_epi32(v_b, _MM_SHUFFLE(1,0,3,2))),
          _mm_cmpeq_epi32(v_a, _mm_shuffle_epi32(v_b, _MM_SHUFFLE(2,1,0,3)))
        )
      ));

      __m128i p = _mm_shuffle_epi8(v_a, shuffle_mask_4x32[mask]);
      _mm_storeu_si128((__m128i*)(out + size_out), p);
      size_out += _mm_popcnt_u32(unsigned(mask));
    } else if (sizeof(T) == 8) {
      int mask = _mm_movemask_pd((__m128d)_mm_or_si128(
        _mm_cmpeq_epi64(v_a, v_b),
        _mm_cmpeq_epi64(v_a, _mm_shuffle_epi32(v_b, _MM_SHUFFLE(1,0,3,2)))
      ));

      __m128i p = _mm_shuffle_epi8(v_a, shuffle_mask_2x64[mask]);
      _mm_storeu_si128((__m128i*)(out + size_out), p);
      size_out += _mm_popcnt_u32(unsigned(mask));
    } else {
      abort();
    }
  }

  return size_out + intersect_scalar(set_a + i, size_a - i, set_b + j, size_b - j, out + size_out);
}

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
template <typename T, typename SIZE_T, typename>
SIZE_T intersect_simd_shuffle_x2(const T *set_a, const SIZE_T size_a, const T *set_b, const SIZE_T size_b, T *out) {
  constexpr static size_t veclen = sizeof(__m128i) / sizeof(T);

  const SIZE_T qs_a = size_a & ~(veclen * 2 - 1);
  const SIZE_T qs_b = size_b & ~(veclen * 2 - 1);
  SIZE_T i = 0, j = 0, size_out = 0;

  while (i < qs_a && j < qs_b) {
    const __m128i v_a0 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_a + i));
    const __m128i v_a1 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_a + i + veclen));
    const __m128i v_b0 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_b + j));
    const __m128i v_b1 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_b + j + veclen));

    const T a_max = set_a[i + veclen * 2 - 1];
    const T b_max = set_b[j + veclen * 2 - 1];
    if (a_max == b_max) {
      i += veclen * 2;
      j += veclen * 2;
      _mm_prefetch(set_a + i, _MM_HINT_NTA);
      _mm_prefetch(set_b + j, _MM_HINT_NTA);
    } else if (a_max < b_max) {
      i += veclen * 2;
      _mm_prefetch(set_a + i, _MM_HINT_NTA);
    } else {
      j += veclen * 2;
      _mm_prefetch(set_b + j, _MM_HINT_NTA);
    }

    if (sizeof(T) == 4) {
      {
        // a0 -- b0/b1
        int mask = _mm_movemask_ps((__m128)_mm_or_si128(
          _mm_or_si128(
            _mm_or_si128(
              _mm_cmpeq_epi32(v_a0, v_b0),
              _mm_cmpeq_epi32(v_a0, _mm_shuffle_epi32(v_b0, _MM_SHUFFLE(0,3,2,1)))
            ),
            _mm_or_si128(
              _mm_cmpeq_epi32(v_a0, _mm_shuffle_epi32(v_b0, _MM_SHUFFLE(1,0,3,2))),
              _mm_cmpeq_epi32(v_a0, _mm_shuffle_epi32(v_b0, _MM_SHUFFLE(2,1,0,3)))
            )
          ),
          _mm_or_si128(
            _mm_or_si128(
              _mm_cmpeq_epi32(v_a0, v_b1),
              _mm_cmpeq_epi32(v_a0, _mm_shuffle_epi32(v_b1, _MM_SHUFFLE(0,3,2,1)))
            ),
            _mm_or_si128(
              _mm_cmpeq_epi32(v_a0, _mm_shuffle_epi32(v_b1, _MM_SHUFFLE(1,0,3,2))),
              _mm_cmpeq_epi32(v_a0, _mm_shuffle_epi32(v_b1, _MM_SHUFFLE(2,1,0,3)))
            )
          )
        ));

        __m128i p = _mm_shuffle_epi8(v_a0, shuffle_mask_4x32[mask]);
        _mm_storeu_si128((__m128i*)(out + size_out), p);
        size_out += _mm_popcnt_u32(unsigned(mask));
      }

      {
        // a1 -- b0/b1
        int mask = _mm_movemask_ps((__m128)_mm_or_si128(
          _mm_or_si128(
            _mm_or_si128(
              _mm_cmpeq_epi32(v_a1, v_b0),
              _mm_cmpeq_epi32(v_a1, _mm_shuffle_epi32(v_b0, _MM_SHUFFLE(0,3,2,1)))
            ),
            _mm_or_si128(
              _mm_cmpeq_epi32(v_a1, _mm_shuffle_epi32(v_b0, _MM_SHUFFLE(1,0,3,2))),
              _mm_cmpeq_epi32(v_a1, _mm_shuffle_epi32(v_b0, _MM_SHUFFLE(2,1,0,3)))
            )
          ),
          _mm_or_si128(
            _mm_or_si128(
              _mm_cmpeq_epi32(v_a1, v_b1),
              _mm_cmpeq_epi32(v_a1, _mm_shuffle_epi32(v_b1, _MM_SHUFFLE(0,3,2,1)))
            ),
            _mm_or_si128(
              _mm_cmpeq_epi32(v_a1, _mm_shuffle_epi32(v_b1, _MM_SHUFFLE(1,0,3,2))),
              _mm_cmpeq_epi32(v_a1, _mm_shuffle_epi32(v_b1, _MM_SHUFFLE(2,1,0,3)))
            )
          )
        ));

        __m128i p = _mm_shuffle_epi8(v_a1, shuffle_mask_4x32[mask]);
        _mm_storeu_si128((__m128i*)(out + size_out), p);
        size_out += _mm_popcnt_u32(unsigned(mask));
      }
    } else if (sizeof(T) == 8) {
      {
        // a0 -- b0/b1
        int mask = _mm_movemask_pd((__m128d)_mm_or_si128(
          _mm_or_si128(
            _mm_cmpeq_epi64(v_a0, v_b0),
            _mm_cmpeq_epi64(v_a0, _mm_shuffle_epi32(v_b0, _MM_SHUFFLE(1,0,3,2)))
          ),
          _mm_or_si128(
            _mm_cmpeq_epi64(v_a0, v_b1),
            _mm_cmpeq_epi64(v_a0, _mm_shuffle_epi32(v_b1, _MM_SHUFFLE(1,0,3,2)))
          )
        ));

        __m128i p = _mm_shuffle_epi8(v_a0, shuffle_mask_2x64[mask]);
        _mm_storeu_si128((__m128i*)(out + size_out), p);
        size_out += _mm_popcnt_u32(unsigned(mask));
      }

      {
        // a1 -- b0/b1
        int mask = _mm_movemask_pd((__m128d)_mm_or_si128(
          _mm_or_si128(
            _mm_cmpeq_epi64(v_a1, v_b0),
            _mm_cmpeq_epi64(v_a1, _mm_shuffle_epi32(v_b0, _MM_SHUFFLE(1,0,3,2)))
          ),
          _mm_or_si128(
            _mm_cmpeq_epi64(v_a1, v_b1),
            _mm_cmpeq_epi64(v_a1, _mm_shuffle_epi32(v_b1, _MM_SHUFFLE(1,0,3,2)))
          )
        ));

        __m128i p = _mm_shuffle_epi8(v_a1, shuffle_mask_2x64[mask]);
        _mm_storeu_si128((__m128i*)(out + size_out), p);
        size_out += _mm_popcnt_u32(unsigned(mask));
      }
    } else {
      abort();
    }
  }

  return size_out + intersect_simd_shuffle(set_a + i, size_a - i, set_b + j, size_b - j, out + size_out);
}

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
template <typename T, typename SIZE_T, typename>
SIZE_T intersect_simd_sttni(const T *set_a, const SIZE_T size_a, const T *set_b, const SIZE_T size_b, T *out) {
  constexpr static size_t veclen = sizeof(__m128i) / sizeof(T);

  const SIZE_T qs_a = size_a - (size_a & (veclen - 1));
  const SIZE_T qs_b = size_b - (size_b & (veclen - 1));
  SIZE_T i = 0, j = 0, size_out = 0;

  while (i < qs_a && j < qs_b) {
    const __m128i v_a = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_a + i));
    const __m128i v_b = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_b + j));

    const T a_max = set_a[i + veclen - 1];
    const T b_max = set_b[j + veclen - 1];
    if (a_max == b_max) {
      i += veclen;
      j += veclen;
      _mm_prefetch(set_a + i, _MM_HINT_NTA);
      _mm_prefetch(set_b + j, _MM_HINT_NTA);
    } else if (a_max < b_max) {
      i += veclen;
      _mm_prefetch(set_a + i, _MM_HINT_NTA);
    } else {
      j += veclen;
      _mm_prefetch(set_b + j, _MM_HINT_NTA);
    }

    int mask = _mm_extract_epi32(_mm_cmpestrm(v_b, 8, v_a, 8, _SIDD_UWORD_OPS|_SIDD_CMP_EQUAL_ANY|_SIDD_BIT_MASK), 0);
    __m128i p = _mm_shuffle_epi8(v_a, shuffle_mask_8x16[mask]);
    _mm_storeu_si128((__m128i*)(out + size_out), p);
    size_out += _mm_popcnt_u32(unsigned(mask));
  }

  return size_out + intersect_scalar(set_a + i, size_a - i, set_b + j, size_b - j, out + size_out);
}

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
template <typename T, typename SIZE_T, typename>
SIZE_T intersect_simd_sttni_x2(const T *set_a, const SIZE_T size_a, const T *set_b, const SIZE_T size_b, T *out) {
  constexpr static size_t veclen = sizeof(__m128i) / sizeof(T);

  const SIZE_T qs_a = size_a - (size_a & (veclen * 2 - 1));
  const SIZE_T qs_b = size_b - (size_b & (veclen * 2 - 1));
  SIZE_T i = 0, j = 0, size_out = 0;

  while (i < qs_a && j < qs_b) {
    const __m128i v_a0 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_a + i));
    const __m128i v_a1 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_a + i + veclen));
    const __m128i v_b0 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_b + j));
    const __m128i v_b1 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(set_b + j + veclen));

    const T a_max = set_a[i + veclen * 2 - 1];
    const T b_max = set_b[j + veclen * 2 - 1];
    if (a_max == b_max) {
      i += veclen * 2;
      j += veclen * 2;
      _mm_prefetch(set_a + i, _MM_HINT_NTA);
      _mm_prefetch(set_b + j, _MM_HINT_NTA);
    } else if (a_max < b_max) {
      i += veclen * 2;
      _mm_prefetch(set_a + i, _MM_HINT_NTA);
    } else {
      j += veclen * 2;
      _mm_prefetch(set_b + j, _MM_HINT_NTA);
    }

    {
      // a0 -- b0
      int mask = _mm_extract_epi32(_mm_cmpestrm(v_b0, 8, v_a0, 8, _SIDD_UWORD_OPS|_SIDD_CMP_EQUAL_ANY|_SIDD_BIT_MASK), 0);
      __m128i p = _mm_shuffle_epi8(v_a0, shuffle_mask_8x16[mask]);
      _mm_storeu_si128((__m128i*)(out + size_out), p);
      size_out += _mm_popcnt_u32(unsigned(mask));
    }

    {
      // a0 -- b1
      int mask = _mm_extract_epi32(_mm_cmpestrm(v_b1, 8, v_a0, 8, _SIDD_UWORD_OPS|_SIDD_CMP_EQUAL_ANY|_SIDD_BIT_MASK), 0);
      __m128i p = _mm_shuffle_epi8(v_a0, shuffle_mask_8x16[mask]);
      _mm_storeu_si128((__m128i*)(out + size_out), p);
      size_out += _mm_popcnt_u32(unsigned(mask));
    }

    {
      // a1 -- b0
      int mask = _mm_extract_epi32(_mm_cmpestrm(v_b0, 8, v_a1, 8, _SIDD_UWORD_OPS|_SIDD_CMP_EQUAL_ANY|_SIDD_BIT_MASK), 0);
      __m128i p = _mm_shuffle_epi8(v_a1, shuffle_mask_8x16[mask]);
      _mm_storeu_si128((__m128i*)(out + size_out), p);
      size_out += _mm_popcnt_u32(unsigned(mask));
    }

    {
      // a1 -- b1
      int mask = _mm_extract_epi32(_mm_cmpestrm(v_b1, 8, v_a1, 8, _SIDD_UWORD_OPS|_SIDD_CMP_EQUAL_ANY|_SIDD_BIT_MASK), 0);
      __m128i p = _mm_shuffle_epi8(v_a1, shuffle_mask_8x16[mask]);
      _mm_storeu_si128((__m128i*)(out + size_out), p);
      size_out += _mm_popcnt_u32(unsigned(mask));
    }
  }

  return size_out + intersect_simd_sttni(set_a + i, size_a - i, set_b + j, size_b - j, out + size_out);
}

struct intersect_simd_helper {
  /**
   * @brief
   * @tparam T
   * @tparam SIZE_T
   * @param set_a
   * @param size_a
   * @param set_b
   * @param size_b
   * @param out
   * @return
   */
  template <typename T, typename SIZE_T>
  static SIZE_T do_intersect_simd(
    const T *set_a, const SIZE_T size_a, const T *set_b, const SIZE_T size_b, T *out,
    typename std::enable_if<sizeof(T) == 2>::type* = nullptr) {
    return intersect_simd_sttni(set_a, size_a, set_b, size_b, out);
  }

  /**
   * @brief
   * @tparam T
   * @tparam SIZE_T
   * @param set_a
   * @param size_a
   * @param set_b
   * @param size_b
   * @param out
   * @return
   */
  template <typename T, typename SIZE_T>
  static SIZE_T do_intersect_simd(
    const T *set_a, const SIZE_T size_a, const T *set_b, const SIZE_T size_b, T *out,
    typename std::enable_if<sizeof(T) == 4 || sizeof(T) == 8>::type* = nullptr) {
    if ((size_a << 3) > size_b && (size_b << 3) > size_a) {
      return intersect_simd_shuffle_x2(set_a, size_a, set_b, size_b, out);
    } else {
      return intersect_simd_galloping(set_a, size_a, set_b, size_b, out);
    }
  }
};

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
template <typename T, typename SIZE_T, typename>
inline
SIZE_T intersect_simd(const T *set_a, const SIZE_T size_a, const T *set_b, const SIZE_T size_b, T *out) {
  return intersect_simd_helper::do_intersect_simd(set_a, size_a, set_b, size_b, out);
}

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
template <typename T, typename SIZE_T, typename>
SIZE_T intersect_simd_shuffle_avx(const T *set_a, const SIZE_T size_a, const T *set_b, const SIZE_T size_b, T *out) {
  constexpr static size_t veclen = sizeof(__m256i) / sizeof(T);
  SIZE_T qs_a = size_a - (size_a & (veclen - 1));
  SIZE_T qs_b = size_b - (size_b & (veclen - 1));
  SIZE_T i = 0, j = 0, size_out = 0;

  while (i < qs_a && j < qs_b) {
    const __m256i v_a = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(set_a + i));
    const __m256i v_b = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(set_b + j));

    const T a_max = set_a[i + veclen - 1];
    const T b_max = set_b[j + veclen - 1];
    if (a_max == b_max) {
      i += veclen;
      j += veclen;
      _mm_prefetch(set_a + i, _MM_HINT_NTA);
      _mm_prefetch(set_b + j, _MM_HINT_NTA);
    } else if (a_max < b_max) {
      i += veclen;
      _mm_prefetch(set_a + i, _MM_HINT_NTA);
    } else {
      j += veclen;
      _mm_prefetch(set_b + j, _MM_HINT_NTA);
    }

    if (sizeof(T) == 4) {
      const __m256i v_b_swap = _mm256_permute4x64_epi64(v_b, _MM_SHUFFLE(1,0,3,2));

      int mask = _mm256_movemask_ps((__m256)_mm256_or_si256(
        _mm256_or_si256(
          _mm256_or_si256(
            _mm256_cmpeq_epi32(v_a, v_b),
            _mm256_cmpeq_epi32(v_a, _mm256_shuffle_epi32(v_b, _MM_SHUFFLE(0,3,2,1)))
          ),
          _mm256_or_si256(
            _mm256_cmpeq_epi32(v_a, _mm256_shuffle_epi32(v_b, _MM_SHUFFLE(1,0,3,2))),
            _mm256_cmpeq_epi32(v_a, _mm256_shuffle_epi32(v_b, _MM_SHUFFLE(2,1,0,3)))
          )
        ),
        _mm256_or_si256(
          _mm256_or_si256(
            _mm256_cmpeq_epi32(v_a, v_b_swap),
            _mm256_cmpeq_epi32(v_a, _mm256_shuffle_epi32(v_b_swap, _MM_SHUFFLE(0,3,2,1)))
          ),
          _mm256_or_si256(
            _mm256_cmpeq_epi32(v_a, _mm256_shuffle_epi32(v_b_swap, _MM_SHUFFLE(1,0,3,2))),
            _mm256_cmpeq_epi32(v_a, _mm256_shuffle_epi32(v_b_swap, _MM_SHUFFLE(2,1,0,3)))
          )
        )));

      __m256i p = _mm256_permutevar8x32_epi32(v_a, shuffle_mask_8x32[mask]);
      _mm256_storeu_si256((__m256i*)(out + size_out), p);
      size_out += _mm_popcnt_u32(unsigned(mask));
    } else if (sizeof(T) == 8) {
      int mask = _mm256_movemask_pd((__m256d)_mm256_or_si256(
        _mm256_or_si256(
          _mm256_cmpeq_epi64(v_a, v_b),
          _mm256_cmpeq_epi64(v_a, _mm256_permute4x64_epi64(v_b, _MM_SHUFFLE(0,3,2,1)))
        ),
        _mm256_or_si256(
          _mm256_cmpeq_epi64(v_a, _mm256_permute4x64_epi64(v_b, _MM_SHUFFLE(1,0,3,2))),
          _mm256_cmpeq_epi64(v_a, _mm256_permute4x64_epi64(v_b, _MM_SHUFFLE(2,1,0,3)))
        )
      ));

      __m256i p = _mm256_permutevar8x32_epi32(v_a, shuffle_mask_4x64[mask]);
      _mm256_storeu_si256((__m256i*)(out + size_out), p);
      size_out += _mm_popcnt_u32(unsigned(mask));
    } else {
      abort();
    }
  }

  return size_out + intersect_simd_shuffle(set_a + i, size_a - i, set_b + j, size_b - j, out + size_out);
}

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
template <typename T, typename SIZE_T, typename>
SIZE_T intersect_simd_shuffle_avx_x2(const T *set_a, const SIZE_T size_a, const T *set_b, const SIZE_T size_b, T *out) {
  constexpr static size_t veclen = sizeof(__m256i) / sizeof(T);
  const SIZE_T qs_a = size_a & ~(veclen * 2 - 1);
  const SIZE_T qs_b = size_b & ~(veclen * 2 - 1);
  SIZE_T i = 0, j = 0, size_out = 0;

  while (i < qs_a && j < qs_b) {
    const __m256i v_a0 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(set_a + i));
    const __m256i v_b0 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(set_b + j));
    const __m256i v_a1 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(set_a + i + veclen));
    const __m256i v_b1 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(set_b + j + veclen));

    const T a_max = set_a[i + veclen * 2 - 1];
    const T b_max = set_b[j + veclen * 2 - 1];

    if (a_max == b_max) {
      i += veclen * 2;
      j += veclen * 2;
      _mm_prefetch(set_a + i, _MM_HINT_NTA);
      _mm_prefetch(set_b + j, _MM_HINT_NTA);
    } else if (a_max < b_max) {
      i += veclen * 2;
      _mm_prefetch(set_a + i, _MM_HINT_NTA);
    } else {
      j += veclen * 2;
      _mm_prefetch(set_b + j, _MM_HINT_NTA);
    }

    if (sizeof(T) == 4) {
      const __m256i v_b0_swap = _mm256_permute4x64_epi64(v_b0, _MM_SHUFFLE(1,0,3,2));
      const __m256i v_b1_swap = _mm256_permute4x64_epi64(v_b1, _MM_SHUFFLE(1,0,3,2));
      {
        // a0 -- b0/b1
        int mask = _mm256_movemask_ps((__m256)_mm256_or_si256(
          _mm256_or_si256(
            _mm256_or_si256(
              _mm256_or_si256(
                _mm256_cmpeq_epi32(v_a0, v_b0),
                _mm256_cmpeq_epi32(v_a0, _mm256_shuffle_epi32(v_b0, _MM_SHUFFLE(0,3,2,1)))
              ),
              _mm256_or_si256(
                _mm256_cmpeq_epi32(v_a0, _mm256_shuffle_epi32(v_b0, _MM_SHUFFLE(1,0,3,2))),
                _mm256_cmpeq_epi32(v_a0, _mm256_shuffle_epi32(v_b0, _MM_SHUFFLE(2,1,0,3)))
              )
            ),
            _mm256_or_si256(
              _mm256_or_si256(
                _mm256_cmpeq_epi32(v_a0, v_b0_swap),
                _mm256_cmpeq_epi32(v_a0, _mm256_shuffle_epi32(v_b0_swap, _MM_SHUFFLE(0,3,2,1)))
              ),
              _mm256_or_si256(
                _mm256_cmpeq_epi32(v_a0, _mm256_shuffle_epi32(v_b0_swap, _MM_SHUFFLE(1,0,3,2))),
                _mm256_cmpeq_epi32(v_a0, _mm256_shuffle_epi32(v_b0_swap, _MM_SHUFFLE(2,1,0,3)))
              )
            )),
          _mm256_or_si256(
            _mm256_or_si256(
              _mm256_or_si256(
                _mm256_cmpeq_epi32(v_a0, v_b1),
                _mm256_cmpeq_epi32(v_a0, _mm256_shuffle_epi32(v_b1, _MM_SHUFFLE(0,3,2,1)))
              ),
              _mm256_or_si256(
                _mm256_cmpeq_epi32(v_a0, _mm256_shuffle_epi32(v_b1, _MM_SHUFFLE(1,0,3,2))),
                _mm256_cmpeq_epi32(v_a0, _mm256_shuffle_epi32(v_b1, _MM_SHUFFLE(2,1,0,3)))
              )
            ),
            _mm256_or_si256(
              _mm256_or_si256(
                _mm256_cmpeq_epi32(v_a0, v_b1_swap),
                _mm256_cmpeq_epi32(v_a0, _mm256_shuffle_epi32(v_b1_swap, _MM_SHUFFLE(0,3,2,1)))
              ),
              _mm256_or_si256(
                _mm256_cmpeq_epi32(v_a0, _mm256_shuffle_epi32(v_b1_swap, _MM_SHUFFLE(1,0,3,2))),
                _mm256_cmpeq_epi32(v_a0, _mm256_shuffle_epi32(v_b1_swap, _MM_SHUFFLE(2,1,0,3)))
              )
            ))));

        __m256i p = _mm256_permutevar8x32_epi32(v_a0, shuffle_mask_8x32[mask]);
        _mm256_storeu_si256((__m256i*)(out + size_out), p);
        size_out += _mm_popcnt_u32(unsigned(mask));
      }

      {
        // a1 -- b0/b1
        int mask = _mm256_movemask_ps((__m256)_mm256_or_si256(
          _mm256_or_si256(
            _mm256_or_si256(
              _mm256_or_si256(
                _mm256_cmpeq_epi32(v_a1, v_b0),
                _mm256_cmpeq_epi32(v_a1, _mm256_shuffle_epi32(v_b0, _MM_SHUFFLE(0,3,2,1)))
              ),
              _mm256_or_si256(
                _mm256_cmpeq_epi32(v_a1, _mm256_shuffle_epi32(v_b0, _MM_SHUFFLE(1,0,3,2))),
                _mm256_cmpeq_epi32(v_a1, _mm256_shuffle_epi32(v_b0, _MM_SHUFFLE(2,1,0,3)))
              )
            ),
            _mm256_or_si256(
              _mm256_or_si256(
                _mm256_cmpeq_epi32(v_a1, v_b0_swap),
                _mm256_cmpeq_epi32(v_a1, _mm256_shuffle_epi32(v_b0_swap, _MM_SHUFFLE(0,3,2,1)))
              ),
              _mm256_or_si256(
                _mm256_cmpeq_epi32(v_a1, _mm256_shuffle_epi32(v_b0_swap, _MM_SHUFFLE(1,0,3,2))),
                _mm256_cmpeq_epi32(v_a1, _mm256_shuffle_epi32(v_b0_swap, _MM_SHUFFLE(2,1,0,3)))
              )
            )),
          _mm256_or_si256(
            _mm256_or_si256(
              _mm256_or_si256(
                _mm256_cmpeq_epi32(v_a1, v_b1),
                _mm256_cmpeq_epi32(v_a1, _mm256_shuffle_epi32(v_b1, _MM_SHUFFLE(0,3,2,1)))
              ),
              _mm256_or_si256(
                _mm256_cmpeq_epi32(v_a1, _mm256_shuffle_epi32(v_b1, _MM_SHUFFLE(1,0,3,2))),
                _mm256_cmpeq_epi32(v_a1, _mm256_shuffle_epi32(v_b1, _MM_SHUFFLE(2,1,0,3)))
              )
            ),
            _mm256_or_si256(
              _mm256_or_si256(
                _mm256_cmpeq_epi32(v_a1, v_b1_swap),
                _mm256_cmpeq_epi32(v_a1, _mm256_shuffle_epi32(v_b1_swap, _MM_SHUFFLE(0,3,2,1)))
              ),
              _mm256_or_si256(
                _mm256_cmpeq_epi32(v_a1, _mm256_shuffle_epi32(v_b1_swap, _MM_SHUFFLE(1,0,3,2))),
                _mm256_cmpeq_epi32(v_a1, _mm256_shuffle_epi32(v_b1_swap, _MM_SHUFFLE(2,1,0,3)))
              )
            ))));

        __m256i p = _mm256_permutevar8x32_epi32(v_a1, shuffle_mask_8x32[mask]);
        _mm256_storeu_si256((__m256i*)(out + size_out), p);
        size_out += _mm_popcnt_u32(unsigned(mask));
      }
    } else if (sizeof(T) == 8) {
      {
        // a0 -- b0/b1
        int mask = _mm256_movemask_pd((__m256d)_mm256_or_si256(
          _mm256_or_si256(
            _mm256_or_si256(
              _mm256_cmpeq_epi64(v_a0, v_b0),
              _mm256_cmpeq_epi64(v_a0, _mm256_permute4x64_epi64(v_b0, _MM_SHUFFLE(0,3,2,1)))
            ),
            _mm256_or_si256(
              _mm256_cmpeq_epi64(v_a0, _mm256_permute4x64_epi64(v_b0, _MM_SHUFFLE(1,0,3,2))),
              _mm256_cmpeq_epi64(v_a0, _mm256_permute4x64_epi64(v_b0, _MM_SHUFFLE(2,1,0,3)))
            )
          ),
          _mm256_or_si256(
            _mm256_or_si256(
              _mm256_cmpeq_epi64(v_a0, v_b1),
              _mm256_cmpeq_epi64(v_a0, _mm256_permute4x64_epi64(v_b1, _MM_SHUFFLE(0,3,2,1)))
            ),
            _mm256_or_si256(
              _mm256_cmpeq_epi64(v_a0, _mm256_permute4x64_epi64(v_b1, _MM_SHUFFLE(1,0,3,2))),
              _mm256_cmpeq_epi64(v_a0, _mm256_permute4x64_epi64(v_b1, _MM_SHUFFLE(2,1,0,3)))
            )
          )
        ));

        __m256i p = _mm256_permutevar8x32_epi32(v_a0, shuffle_mask_4x64[mask]);
        _mm256_storeu_si256((__m256i*)(out + size_out), p);
        size_out += _mm_popcnt_u32(unsigned(mask));
      }

      {
        // a1 -- b0/b1
        int mask = _mm256_movemask_pd((__m256d)_mm256_or_si256(
          _mm256_or_si256(
            _mm256_or_si256(
              _mm256_cmpeq_epi64(v_a1, v_b0),
              _mm256_cmpeq_epi64(v_a1, _mm256_permute4x64_epi64(v_b0, _MM_SHUFFLE(0,3,2,1)))
            ),
            _mm256_or_si256(
              _mm256_cmpeq_epi64(v_a1, _mm256_permute4x64_epi64(v_b0, _MM_SHUFFLE(1,0,3,2))),
              _mm256_cmpeq_epi64(v_a1, _mm256_permute4x64_epi64(v_b0, _MM_SHUFFLE(2,1,0,3)))
            )
          ),
          _mm256_or_si256(
            _mm256_or_si256(
              _mm256_cmpeq_epi64(v_a1, v_b1),
              _mm256_cmpeq_epi64(v_a1, _mm256_permute4x64_epi64(v_b1, _MM_SHUFFLE(0,3,2,1)))
            ),
            _mm256_or_si256(
              _mm256_cmpeq_epi64(v_a1, _mm256_permute4x64_epi64(v_b1, _MM_SHUFFLE(1,0,3,2))),
              _mm256_cmpeq_epi64(v_a1, _mm256_permute4x64_epi64(v_b1, _MM_SHUFFLE(2,1,0,3)))
            )
          )
        ));

        __m256i p = _mm256_permutevar8x32_epi32(v_a1, shuffle_mask_4x64[mask]);
        _mm256_storeu_si256((__m256i*)(out + size_out), p);
        size_out += _mm_popcnt_u32(unsigned(mask));
      }
    } else {
      abort();
    }
  }

  return size_out + intersect_simd_shuffle(set_a + i, size_a - i, set_b + j, size_b - j, out + size_out);
}

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
template <typename T, typename SIZE_T, typename>
SIZE_T intersect_simd_galloping_avx(const T *set_a, const SIZE_T size_a, const T *set_b, const SIZE_T size_b, T *out) {
  constexpr static size_t veclen = sizeof(__m256i) / sizeof(T);
  const T *set_freq = size_a < size_b ? set_b : set_a;
  const SIZE_T size_freq = size_a < size_b ? size_b : size_a;
  const T *set_rare = size_a < size_b ? set_a : set_b;
  const SIZE_T size_rare = size_a < size_b ? size_a : size_b;

  const SIZE_T qs_freq = size_freq & ~(veclen * 16 - 1);
  const SIZE_T qs_rare = size_rare;
  SIZE_T i = 0, j = 0, size_out = 0;

  while (i < qs_freq && j < qs_rare) {
    const T match_rare = set_rare[j];
    if (set_freq[i + veclen * 16 - 1] < set_rare[j]) {
      i += veclen * 16;
      continue;
    }

    __m256i test;

    if (sizeof(T) == 4) {
      const int match_rare_tmp = match_rare;
      const __m256i match = _mm256_set1_epi32(match_rare_tmp);
      if (set_freq[i + veclen * 8 - 1] < match_rare) {
        test = _mm256_or_si256(
          _mm256_or_si256(
            _mm256_or_si256(
              _mm256_cmpeq_epi32(_mm256_lddqu_si256(reinterpret_cast<const __m256i *>(set_freq + i) + 8), match),
              _mm256_cmpeq_epi32(_mm256_lddqu_si256(reinterpret_cast<const __m256i *>(set_freq + i) + 9), match)
            ),
            _mm256_or_si256(
              _mm256_cmpeq_epi32(_mm256_lddqu_si256(reinterpret_cast<const __m256i *>(set_freq + i) + 10), match),
              _mm256_cmpeq_epi32(_mm256_lddqu_si256(reinterpret_cast<const __m256i *>(set_freq + i) + 11), match)
            )),
          _mm256_or_si256(
            _mm256_or_si256(
              _mm256_cmpeq_epi32(_mm256_lddqu_si256(reinterpret_cast<const __m256i *>(set_freq + i) + 12), match),
              _mm256_cmpeq_epi32(_mm256_lddqu_si256(reinterpret_cast<const __m256i *>(set_freq + i) + 13), match)
            ),
            _mm256_or_si256(
              _mm256_cmpeq_epi32(_mm256_lddqu_si256(reinterpret_cast<const __m256i *>(set_freq + i) + 14), match),
              _mm256_cmpeq_epi32(_mm256_lddqu_si256(reinterpret_cast<const __m256i *>(set_freq + i) + 15), match)
            )
          )
        );
      } else {
        test = _mm256_or_si256(
          _mm256_or_si256(
            _mm256_or_si256(
              _mm256_cmpeq_epi32(_mm256_lddqu_si256(reinterpret_cast<const __m256i *>(set_freq + i) + 0), match),
              _mm256_cmpeq_epi32(_mm256_lddqu_si256(reinterpret_cast<const __m256i *>(set_freq + i) + 1), match)
            ),
            _mm256_or_si256(
              _mm256_cmpeq_epi32(_mm256_lddqu_si256(reinterpret_cast<const __m256i *>(set_freq + i) + 2), match),
              _mm256_cmpeq_epi32(_mm256_lddqu_si256(reinterpret_cast<const __m256i *>(set_freq + i) + 3), match)
            )),
          _mm256_or_si256(
            _mm256_or_si256(
              _mm256_cmpeq_epi32(_mm256_lddqu_si256(reinterpret_cast<const __m256i *>(set_freq + i) + 4), match),
              _mm256_cmpeq_epi32(_mm256_lddqu_si256(reinterpret_cast<const __m256i *>(set_freq + i) + 5), match)
            ),
            _mm256_or_si256(
              _mm256_cmpeq_epi32(_mm256_lddqu_si256(reinterpret_cast<const __m256i *>(set_freq + i) + 6), match),
              _mm256_cmpeq_epi32(_mm256_lddqu_si256(reinterpret_cast<const __m256i *>(set_freq + i) + 7), match)
            )
          )
        );
      }
    } else if (sizeof(T) == 8) {
      const long long match_rare_tmp = match_rare;
      __m256i match = _mm256_set1_epi64x(match_rare_tmp);
      if (set_freq[i + veclen * 8 - 1] < match_rare) {
        test = _mm256_or_si256(
          _mm256_or_si256(
            _mm256_or_si256(
              _mm256_cmpeq_epi64(_mm256_lddqu_si256(reinterpret_cast<const __m256i *>(set_freq + i) + 8), match),
              _mm256_cmpeq_epi64(_mm256_lddqu_si256(reinterpret_cast<const __m256i *>(set_freq + i) + 9), match)
            ),
            _mm256_or_si256(
              _mm256_cmpeq_epi64(_mm256_lddqu_si256(reinterpret_cast<const __m256i *>(set_freq + i) + 10), match),
              _mm256_cmpeq_epi64(_mm256_lddqu_si256(reinterpret_cast<const __m256i *>(set_freq + i) + 11), match)
            )),
          _mm256_or_si256(
            _mm256_or_si256(
              _mm256_cmpeq_epi64(_mm256_lddqu_si256(reinterpret_cast<const __m256i *>(set_freq + i) + 12), match),
              _mm256_cmpeq_epi64(_mm256_lddqu_si256(reinterpret_cast<const __m256i *>(set_freq + i) + 13), match)
            ),
            _mm256_or_si256(
              _mm256_cmpeq_epi64(_mm256_lddqu_si256(reinterpret_cast<const __m256i *>(set_freq + i) + 14), match),
              _mm256_cmpeq_epi64(_mm256_lddqu_si256(reinterpret_cast<const __m256i *>(set_freq + i) + 15), match)
            )
          )
        );
      } else {
        test = _mm256_or_si256(
          _mm256_or_si256(
            _mm256_or_si256(
              _mm256_cmpeq_epi64(_mm256_lddqu_si256(reinterpret_cast<const __m256i *>(set_freq + i) + 0), match),
              _mm256_cmpeq_epi64(_mm256_lddqu_si256(reinterpret_cast<const __m256i *>(set_freq + i) + 1), match)
            ),
            _mm256_or_si256(
              _mm256_cmpeq_epi64(_mm256_lddqu_si256(reinterpret_cast<const __m256i *>(set_freq + i) + 2), match),
              _mm256_cmpeq_epi64(_mm256_lddqu_si256(reinterpret_cast<const __m256i *>(set_freq + i) + 3), match)
            )),
          _mm256_or_si256(
            _mm256_or_si256(
              _mm256_cmpeq_epi64(_mm256_lddqu_si256(reinterpret_cast<const __m256i *>(set_freq + i) + 4), match),
              _mm256_cmpeq_epi64(_mm256_lddqu_si256(reinterpret_cast<const __m256i *>(set_freq + i) + 5), match)
            ),
            _mm256_or_si256(
              _mm256_cmpeq_epi64(_mm256_lddqu_si256(reinterpret_cast<const __m256i *>(set_freq + i) + 6), match),
              _mm256_cmpeq_epi64(_mm256_lddqu_si256(reinterpret_cast<const __m256i *>(set_freq + i) + 7), match)
            )
          )
        );
      }
    } else {
      abort();
    }

    if (!_mm256_testz_si256(test, test)) {
      out[size_out++] = set_rare[j];
    }

    ++j;
  }

  return size_out + intersect_simd_shuffle(set_freq + i, size_freq - i, set_rare + j, size_rare - j, out + size_out);
}


struct intersect_simd_avx_helper {
  /**
   * @brief
   * @tparam T
   * @tparam SIZE_T
   * @param set_a
   * @param size_a
   * @param set_b
   * @param size_b
   * @param out
   * @return
   */
  template <typename T, typename SIZE_T>
  static SIZE_T do_intersect_simd(
    const T *set_a, const SIZE_T size_a, const T *set_b, const SIZE_T size_b, T *out,
    typename std::enable_if<sizeof(T) == 2>::type* = nullptr) {
    return intersect_simd_sttni(set_a, size_a, set_b, size_b, out);
  }

  /**
   * @brief
   * @tparam T
   * @tparam SIZE_T
   * @param set_a
   * @param size_a
   * @param set_b
   * @param size_b
   * @param out
   * @return
   */
  template <typename T, typename SIZE_T>
  static SIZE_T do_intersect_simd(
    const T *set_a, const SIZE_T size_a, const T *set_b, const SIZE_T size_b, T *out,
    typename std::enable_if<sizeof(T) == 4 || sizeof(T) == 8>::type* = nullptr) {
    if ((size_a << 3) > size_b && (size_b << 3) > size_a) {
      return intersect_simd_shuffle_avx(set_a, size_a, set_b, size_b, out);
    } else {
      return intersect_simd_galloping_avx(set_a, size_a, set_b, size_b, out);
    }
  }
};

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
template <typename T, typename SIZE_T, typename>
inline
SIZE_T intersect_simd_avx(const T *set_a, const SIZE_T size_a, const T *set_b, const SIZE_T size_b, T *out) {
  return intersect_simd_avx_helper::do_intersect_simd(set_a, size_a, set_b, size_b, out);
}

#endif

#ifdef __USE_AVX512__

template <typename T, typename SIZE_T, typename>
SIZE_T intersect_simd_shuffle_avx512(const T *set_a, const SIZE_T size_a, const T *set_b, const SIZE_T size_b, T *out) {
  constexpr static size_t veclen = sizeof(__m512i) / sizeof(T);
  SIZE_T size_out = 0;
  int64_t i = 0, j = 0;

  if (sizeof(T) == 4) {
    __m512i v_a, v_b;
    __mmask16 a_need_mask = 0xFFFF, b_need_mask = 0xFFFF;
    while (i < int64_t(size_a) - int64_t(veclen) && j < int64_t(size_b) - int64_t(veclen)) {
      v_a = _mm512_mask_expandloadu_epi32(v_a, a_need_mask, set_a + i + veclen - _mm_popcnt_u32(a_need_mask));
      v_b = _mm512_mask_expandloadu_epi32(v_b, b_need_mask, set_b + j + veclen - _mm_popcnt_u32(b_need_mask));
      const T a_max = set_a[i + veclen - 1];
      const T b_max = set_b[j + veclen - 1];
      if (std::is_same<T, uint32_t>::value) {
        a_need_mask = _mm512_cmple_epu32_mask(v_a, _mm512_set1_epi32(b_max));
        b_need_mask = _mm512_cmple_epu32_mask(v_b, _mm512_set1_epi32(a_max));
      } else if (std::is_same<T, int32_t>::value) {
        a_need_mask = _mm512_cmple_epi32_mask(v_a, _mm512_set1_epi32(b_max));
        b_need_mask = _mm512_cmple_epi32_mask(v_b, _mm512_set1_epi32(a_max));
      } else if (std::is_same<T, float>::value) {
        a_need_mask = _mm512_cmple_ps_mask((__m512)v_a, _mm512_set1_ps(b_max));
        b_need_mask = _mm512_cmple_ps_mask((__m512)v_b, _mm512_set1_ps(a_max));
      } else {
        abort();
      }

      i += _mm_popcnt_u32(a_need_mask);
      j += _mm_popcnt_u32(b_need_mask);

      _mm_prefetch(set_a + i, _MM_HINT_T0);
      _mm_prefetch(set_b + j, _MM_HINT_T0);
      _mm_prefetch(set_a + i + veclen, _MM_HINT_T1);
      _mm_prefetch(set_b + j + veclen, _MM_HINT_T1);

      __mmask16 mask =_mm512_kor(
        _mm512_kor(
          _mm512_kor(
            _mm512_kor(
              _mm512_cmpeq_epi32_mask(v_a, v_b),
              _mm512_cmpeq_epi32_mask(v_a, _mm512_permutexvar_epi32((__m512i)(__v16si){1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0}, v_b))
            ),
            _mm512_kor(
              _mm512_cmpeq_epi32_mask(v_a, _mm512_permutexvar_epi32((__m512i)(__v16si){2,3,4,5,6,7,8,9,10,11,12,13,14,15,0,1}, v_b)),
              _mm512_cmpeq_epi32_mask(v_a, _mm512_permutexvar_epi32((__m512i)(__v16si){3,4,5,6,7,8,9,10,11,12,13,14,15,0,1,2}, v_b))
            )
          ),
          _mm512_kor(
            _mm512_kor(
              _mm512_cmpeq_epi32_mask(v_a, _mm512_permutexvar_epi32((__m512i)(__v16si){4,5,6,7,8,9,10,11,12,13,14,15,0,1,2,3}, v_b)),
              _mm512_cmpeq_epi32_mask(v_a, _mm512_permutexvar_epi32((__m512i)(__v16si){5,6,7,8,9,10,11,12,13,14,15,0,1,2,3,4}, v_b))
            ),
            _mm512_kor(
              _mm512_cmpeq_epi32_mask(v_a, _mm512_permutexvar_epi32((__m512i)(__v16si){6,7,8,9,10,11,12,13,14,15,0,1,2,3,4,5}, v_b)),
              _mm512_cmpeq_epi32_mask(v_a, _mm512_permutexvar_epi32((__m512i)(__v16si){7,8,9,10,11,12,13,14,15,0,1,2,3,4,5,6}, v_b))
            )
          )
        ),
        _mm512_kor(
          _mm512_kor(
            _mm512_kor(
              _mm512_cmpeq_epi32_mask(v_a, _mm512_permutexvar_epi32((__m512i)(__v16si){8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7}, v_b)),
              _mm512_cmpeq_epi32_mask(v_a, _mm512_permutexvar_epi32((__m512i)(__v16si){9,10,11,12,13,14,15,0,1,2,3,4,5,6,7,8}, v_b))
            ),
            _mm512_kor(
              _mm512_cmpeq_epi32_mask(v_a, _mm512_permutexvar_epi32((__m512i)(__v16si){10,11,12,13,14,15,0,1,2,3,4,5,6,7,8,9}, v_b)),
              _mm512_cmpeq_epi32_mask(v_a, _mm512_permutexvar_epi32((__m512i)(__v16si){11,12,13,14,15,0,1,2,3,4,5,6,7,8,9,10}, v_b))
            )
          ),
          _mm512_kor(
            _mm512_kor(
              _mm512_cmpeq_epi32_mask(v_a, _mm512_permutexvar_epi32((__m512i)(__v16si){12,13,14,15,0,1,2,3,4,5,6,7,8,9,10,11}, v_b)),
              _mm512_cmpeq_epi32_mask(v_a, _mm512_permutexvar_epi32((__m512i)(__v16si){13,14,15,0,1,2,3,4,5,6,7,8,9,10,11,12}, v_b))
            ),
            _mm512_kor(
              _mm512_cmpeq_epi32_mask(v_a, _mm512_permutexvar_epi32((__m512i)(__v16si){14,15,0,1,2,3,4,5,6,7,8,9,10,11,12,13}, v_b)),
              _mm512_cmpeq_epi32_mask(v_a, _mm512_permutexvar_epi32((__m512i)(__v16si){15,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14}, v_b))
            )
          )
        )
      );
      _mm512_mask_compressstoreu_epi32(out + size_out, mask, v_a);
      size_out += _mm_popcnt_u32(mask);

    }
  } else if (sizeof(T) == 8) {
    __m512i v_a, v_b;
    __mmask8 a_need_mask = 0xFF, b_need_mask = 0xFF;

    while (i < int64_t(size_a) - int64_t(veclen) && j < int64_t(size_b) - int64_t(veclen)) {
      v_a = _mm512_mask_expandloadu_epi64(v_a, a_need_mask, set_a + i + veclen - _mm_popcnt_u32(a_need_mask));
      v_b = _mm512_mask_expandloadu_epi64(v_b, b_need_mask, set_b + j + veclen - _mm_popcnt_u32(b_need_mask));
      const T a_max = set_a[i + veclen - 1];
      const T b_max = set_b[j + veclen - 1];
      if (std::is_same<T, uint32_t>::value) {
        a_need_mask = _mm512_cmple_epu64_mask(v_a, _mm512_set1_epi64(b_max));
        b_need_mask = _mm512_cmple_epu64_mask(v_b, _mm512_set1_epi64(a_max));
      } else if (std::is_same<T, int32_t>::value) {
        a_need_mask = _mm512_cmple_epi64_mask(v_a, _mm512_set1_epi64(b_max));
        b_need_mask = _mm512_cmple_epi64_mask(v_b, _mm512_set1_epi64(a_max));
      } else if (std::is_same<T, float>::value) {
        a_need_mask = _mm512_cmple_pd_mask((__m512d)v_a, _mm512_set1_pd(b_max));
        b_need_mask = _mm512_cmple_pd_mask((__m512d)v_b, _mm512_set1_pd(a_max));
      } else {
        abort();
      }

      i += _mm_popcnt_u32(a_need_mask);
      j += _mm_popcnt_u32(b_need_mask);

      _mm_prefetch(set_a + i, _MM_HINT_T0);
      _mm_prefetch(set_b + j, _MM_HINT_T0);
      _mm_prefetch(set_a + i + veclen, _MM_HINT_T1);
      _mm_prefetch(set_b + j + veclen, _MM_HINT_T1);

      __mmask8 mask = _mm512_kor(
        _mm512_kor(
          _mm512_kor(
            _mm512_cmpeq_epi64_mask(v_a, v_b),
            _mm512_cmpeq_epi64_mask(v_a, _mm512_permutexvar_epi64((__m512i) (__v8di) {1, 2, 3, 4, 5, 6, 7, 0}, v_b))
          ),
          _mm512_kor(
            _mm512_cmpeq_epi64_mask(v_a, _mm512_permutexvar_epi64((__m512i) (__v8di) {2, 3, 4, 5, 6, 7, 0, 1}, v_b)),
            _mm512_cmpeq_epi64_mask(v_a, _mm512_permutexvar_epi64((__m512i) (__v8di) {3, 4, 5, 6, 7, 0, 1, 2}, v_b))
          )
        ),
        _mm512_kor(
          _mm512_kor(
            _mm512_cmpeq_epi64_mask(v_a, _mm512_permutexvar_epi64((__m512i) (__v8di) {4, 5, 6, 7, 0, 1, 2, 3}, v_b)),
            _mm512_cmpeq_epi64_mask(v_a, _mm512_permutexvar_epi64((__m512i) (__v8di) {5, 6, 7, 0, 1, 2, 3, 4}, v_b))
          ),
          _mm512_kor(
            _mm512_cmpeq_epi64_mask(v_a, _mm512_permutexvar_epi64((__m512i) (__v8di) {6, 7, 0, 1, 2, 3, 4, 5}, v_b)),
            _mm512_cmpeq_epi64_mask(v_a, _mm512_permutexvar_epi64((__m512i) (__v8di) {7, 0, 1, 2, 3, 4, 5, 6}, v_b))
          )
        )
      );

      _mm512_mask_compressstoreu_epi64(out + size_out, mask, v_a);
      size_out += _mm_popcnt_u32(mask);
    }
  } else {
    abort();
  }

  return size_out + intersect_scalar(set_a + i, size_a - i, set_b + j, size_b - j, out + size_out);
}

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
template <typename T, typename SIZE_T, typename>
SIZE_T intersect_simd_galloping_avx512(const T *set_a, const SIZE_T size_a, const T *set_b, const SIZE_T size_b, T *out) {
  constexpr static size_t veclen = sizeof(__m512i) / sizeof(T);
  const T *set_freq = size_a < size_b ? set_b : set_a;
  const SIZE_T size_freq = size_a < size_b ? size_b : size_a;
  const T *set_rare = size_a < size_b ? set_a : set_b;
  const SIZE_T size_rare = size_a < size_b ? size_a : size_b;

  const SIZE_T qs_freq = size_freq & ~(veclen * veclen - 1);
  const SIZE_T qs_rare = size_rare;
  SIZE_T i = 0, j = 0, size_out = 0;

  while (i < qs_freq && j < qs_rare) {
    const T match_rare = set_rare[j];
    if (set_freq[i + veclen * veclen - 1] < set_rare[j]) {
      i += veclen * veclen;
      continue;
    }

    bool test;

    if (sizeof(T) == 4) {
      const int match_rare_tmp = match_rare;
      const __m512i match = _mm512_set1_epi32(match_rare_tmp);
      __m512i index = _mm512_i32gather_epi32(
        (__m512i)(__v16si){veclen*0,veclen*1,veclen*2,veclen*3,veclen*4,veclen*5,veclen*6,veclen*7,veclen*8,veclen*9,veclen*10,veclen*11,veclen*12,veclen*13,veclen*14,veclen*15},
        set_freq + i, sizeof(T));

      int offset;
      if (std::is_same<T, uint32_t>::value) {
        offset = _mm_popcnt_u32(_mm512_cmple_epu32_mask(index, match));
      } else if (std::is_same<T, int32_t>::value) {
        offset = _mm_popcnt_u32(_mm512_cmple_epi32_mask(index, match));
      } else if (std::is_same<T, float>::value) {
        offset = _mm_popcnt_u32(_mm512_cmple_ps_mask((__m512)index, (__m512)match));
      } else {
        abort();
      }

      test = _mm512_cmpeq_epi32_mask(_mm512_loadu_si512(set_freq + i + (offset - 1) * veclen), match);
    } else if (sizeof(T) == 8) {
      const long long match_rare_tmp = match_rare;
      __m512i match = _mm512_set1_epi64(match_rare_tmp);
      __m512i index = _mm512_i64gather_epi64(
        (__m512i)(__v8di){veclen*0,veclen*1,veclen*2,veclen*3,veclen*4,veclen*5,veclen*6,veclen*7},
        set_freq + i, sizeof(T));

      int offset;
      if (std::is_same<T, uint64_t>::value) {
        offset = _mm_popcnt_u32(_mm512_cmple_epu64_mask(index, match));
      } else if (std::is_same<T, int64_t>::value) {
        offset = _mm_popcnt_u32(_mm512_cmple_epi64_mask(index, match));
      } else if (std::is_same<T, double>::value) {
        offset = _mm_popcnt_u32(_mm512_cmple_pd_mask((__m512d)index, (__m512d)match));
      } else {
        abort();
      }
      test = _mm512_cmpeq_epi64_mask(_mm512_loadu_si512(set_freq + i + (offset - 1) * veclen), match);
    } else {
      abort();
    }

    if (test) {
      out[size_out++] = set_rare[j];
    }

    ++j;
  }

  return size_out + intersect_scalar(set_freq + i, size_freq - i, set_rare + j, size_rare - j, out + size_out);
}

struct intersect_simd_avx512_helper {
  template <typename T, typename SIZE_T>
  static SIZE_T do_intersect_simd(
    const T *set_a, const SIZE_T size_a, const T *set_b, const SIZE_T size_b, T *out,
    typename std::enable_if<sizeof(T) == 2>::type* = nullptr) {
    return intersect_simd_sttni(set_a, size_a, set_b, size_b, out);
  }

  template <typename T, typename SIZE_T>
  static SIZE_T do_intersect_simd(
    const T *set_a, const SIZE_T size_a, const T *set_b, const SIZE_T size_b, T *out,
    typename std::enable_if<sizeof(T) == 4 || sizeof(T) == 8>::type* = nullptr) {
    if ((size_a << 4) > size_b && (size_b << 4) > size_a) {
      return intersect_simd_shuffle_avx512(set_a, size_a, set_b, size_b, out);
    } else {
      return intersect_simd_galloping_avx512(set_a, size_a, set_b, size_b, out);
    }
  }
};

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
template <typename T, typename SIZE_T, typename>
inline
SIZE_T intersect_simd_avx512(const T *set_a, const SIZE_T size_a, const T *set_b, const SIZE_T size_b, T *out) {
  return intersect_simd_avx512_helper::do_intersect_simd(set_a, size_a, set_b, size_b, out);
}

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
template <typename T, typename SIZE_T, typename>
inline
SIZE_T intersect(const T *set_a, SIZE_T size_a, const T *set_b, SIZE_T size_b, T *out) {
#ifdef __USE_AVX512__
#ifdef __AVX512F__
  return intersect_simd_avx512(set_a, size_a, set_b, size_b, out);
#else
  return cpu_support_avx512 ?
         intersect_simd_avx512(set_a, size_a, set_b, size_b, out) :
         intersect_simd_avx(set_a, size_a, set_b, size_b, out);
#endif
#elif defined(__AVX2__)
  return intersect_simd_avx(set_a, size_a, set_b, size_b, out);
#else
  return intersect_simd(set_a, size_a, set_b, size_b, out);
#endif
}

}