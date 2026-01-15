from laf_asm_code import laf_asm_code
from global_config import *

def gemm_MxNxK_impl(M, N, K, lda, ldb, ldc, uniq_id, repeat, pipeline_strategy_level = 0, UNROLL_K = 8, NR_MAIN = 4, MRSA_FLAG = 0):
    # lda, ldb, ldc = M, K, M # 这里意味着C矩阵是列主序，A矩阵是列主序，B矩阵是列主序
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

extern "C" {{
void small_gemm_nn_with_bias(const float *A, const float *B, float *C, int lda, int ldb, int ldc);
void small_gemm_nt_with_bias(const float *A, const float *B, float *C, int lda, int ldb, int ldc);
void small_gemm_tn_with_bias(const float *A, const float *B, float *C, int lda, int ldb, int ldc);
void small_gemm_tt_with_bias(const float *A, const float *B, float *C, int lda, int ldb, int ldc);
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

  float *A = static_cast<float*>(_mm_malloc(64, M * lda * sizeof(float)));
  float *B = static_cast<float*>(_mm_malloc(64, K * ldb * sizeof(float)));
  float *C = static_cast<float*>(_mm_malloc(64, M * ldc * sizeof(float)));
  float *refC = static_cast<float*>(_mm_malloc(64, M * ldc * sizeof(float)));
  float *ourC = static_cast<float*>(_mm_malloc(64, M * ldc * sizeof(float)));

  test_utils::init(A, M * lda);
  test_utils::init(B, K * ldb);
  test_utils::init(C, M * ldc);

  int n_warming = 20;
  int n_loops = {repeat};

  bool ACC = false;
  for (int i = 0; i < M; ++i) {{
    for (int j = 0; j < N; ++j) {{
      float c = 10.0f * rand() / RAND_MAX;
      refC[i * ldc + j] = c;
      ourC[i * ldc + j] = c;
    }}
  }}
  ACC = true;
  test_utils::gemm_ref(A, B, refC, M, N, K, lda, ldb, ldc, ACC);
  small_gemm_with_bias(A, B, ourC, lda, ldb, ldc);
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