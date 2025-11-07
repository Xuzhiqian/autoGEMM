from laf_asm_code import laf_asm_code
from global_config import *

def gemm_MxKxN_impl(M, K, N, lda, ldb, ldc, uniq_id, repeat, pipeline_strategy_level = 0, UNROLL_K = 8, NR_MAIN = 4):
    # lda, ldb, ldc = K, N, N # 这里意味着C矩阵是行主序，A矩阵是行主序，B矩阵是行主序
    """Emit C code for gemm impl."""
    # 通过laf_asm_code生成了small_gemm接口
    # 通过laf_asm_code生成了small_gemm_with_bias接口
    cc_code = f"""
#ifndef __SGEMM_KERNEL_H
#define __SGEMM_KERNEL_H
#endif
#include <cmath>
#include <cstring>
#include <cassert>
#include <arm_neon.h>
#include <cstdlib>
#include <cstdio>
#include "test.h"
#include "timer.h"

namespace laf {{
void small_gemm(const {DATA_TYPE} *A, const {DATA_TYPE} *B, {DATA_TYPE} *C, int lda, int ldb, int ldc) {{
"""
    cc_code += laf_asm_code(M, N, K, lda, ldb, ldc, pipeline_strategy_level=pipeline_strategy_level, UNROLL_K=UNROLL_K, NR_MAIN=NR_MAIN, with_bias = 0)
    cc_code += f"""
}}
void small_gemm_with_bias(const {DATA_TYPE} *A, const {DATA_TYPE} *B, {DATA_TYPE} *C, int lda, int ldb, int ldc) {{
"""
    cc_code += laf_asm_code(M, N, K, lda, ldb, ldc, pipeline_strategy_level=pipeline_strategy_level, UNROLL_K=UNROLL_K, NR_MAIN=NR_MAIN, with_bias = 1)
    cc_code += f"""
}}
}}

extern "C" int gemm_{M}x{K}x{N}_{lda}_{ldb}_{ldc}_xsmm_{uniq_id}(const {DATA_TYPE} *A, const {DATA_TYPE} *B, {DATA_TYPE} *C, const int lda, const int ldb, const int ldc){{
  laf::small_gemm(A, B, C, lda, ldb, ldc);
  return 0;
}}

extern "C" int gemm_{M}x{K}x{N}_{lda}_{ldb}_{ldc}_xsmm_with_bias_{uniq_id}(const {DATA_TYPE} *A, const {DATA_TYPE} *B, {DATA_TYPE} *C, const int lda, const int ldb, const int ldc){{
  laf::small_gemm_with_bias(A, B, C, lda, ldb, ldc);
  return 0;
}}

void* _mm_malloc(size_t align, size_t sz)
{{
  void *ptr;
  int alloc_result = posix_memalign(&ptr, align, sz);
  if(alloc_result != 0)
  {{
    return NULL;
  }}
  return ptr;
}}

int main() {{
  #define M {M}
  #define N {N}
  #define K {K}

  #define lda {lda}
  #define ldb {ldb}
  #define ldc {ldc}

  {DATA_TYPE} *A = static_cast<{DATA_TYPE}*>(_mm_malloc(64, M * lda * sizeof({DATA_TYPE})));
  {DATA_TYPE} *B = static_cast<{DATA_TYPE}*>(_mm_malloc(64, K * ldb * sizeof({DATA_TYPE})));
  {DATA_TYPE} *C = static_cast<{DATA_TYPE}*>(_mm_malloc(64, M * ldc * sizeof({DATA_TYPE})));
  {DATA_TYPE} *refC = static_cast<{DATA_TYPE}*>(_mm_malloc(64, M * ldc * sizeof({DATA_TYPE})));
  {DATA_TYPE} *ourC = static_cast<{DATA_TYPE}*>(_mm_malloc(64, M * ldc * sizeof({DATA_TYPE})));

  test_utils::init(A, M * lda);
  test_utils::init(B, K * ldb);
  test_utils::init(C, M * ldc);

  int n_warming = 20;
  int n_loops = {repeat};

  for (int i = 0; i < n_warming; ++i) {{
    laf::small_gemm_with_bias(A, B, C, lda, ldb, ldc);
  }}

  Timer t;
  for (int i = 0; i < n_loops; ++i) {{
    laf::small_gemm_with_bias(A, B, C, lda, ldb, ldc);
  }}

  float latency = t.getTime();
  float gflops = M * N * K * 2 / latency * n_loops / 1000000;
  printf("%.2f\t", gflops);

  bool ACC = false;
  test_utils::gemm_ref(A, B, refC, M, N, K, lda, ldb, ldc, ACC);
  laf::small_gemm(A, B, ourC, lda, ldb, ldc);
  if (!test_utils::is_same_matrix(refC, ourC, M, N, ldc, {TOL}, {TOL})) {{
    int idx = test_utils::diff_index(refC, ourC, M, N, ldc, {TOL}, {TOL});
    printf("ERROR: M=%d, N=%d, K=%d, lda=%d, ldb=%d, ldc=%d, ACC=%d, ref[%d]=%.6f, our[%d]=%.6f\\n",
           M, N, K, lda, ldb, ldc, ACC, idx, refC[idx], idx, ourC[idx]);
    test_utils::print_diff(refC, ourC, M, N, ldc);
  }} else {{
    //printf("0------passed\\n");
  }}
  for (int i = 0; i < M; ++i) {{
    for (int j = 0; j < N; ++j) {{
      {DATA_TYPE} c = 10.0f * rand() / RAND_MAX;
      refC[i * ldc + j] = c;
      ourC[i * ldc + j] = c;
    }}
  }}
  ACC = true;
  test_utils::gemm_ref(A, B, refC, M, N, K, lda, ldb, ldc, ACC);
  laf::small_gemm_with_bias(A, B, ourC, lda, ldb, ldc);
  if (!test_utils::is_same_matrix(refC, ourC, M, N, ldc, {TOL}, {TOL})) {{
    int idx = test_utils::diff_index(refC, ourC, M, N, ldc, {TOL}, {TOL});
    printf("ERROR: M=%d, N=%d, K=%d, lda=%d, ldb=%d, ldc=%d, ACC=%d, ref[%d]=%.6f, our[%d]=%.6f\\n",
           M, N, K, lda, ldb, ldc, ACC, idx, refC[idx], idx, ourC[idx]);
    test_utils::print_diff(refC, ourC, M, N, ldc);
  }} else {{
    //printf("1------passed\\n");
  }}

"""
    cc_code += f"""
  free(A);
  A=NULL;
  free(B);
  B=NULL;
  free(C);
  C=NULL;
  free(refC);
  refC=NULL;
  free(ourC);
  ourC=NULL;
}}
"""
    return cc_code