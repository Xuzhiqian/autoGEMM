from laf_asm_code import laf_asm_code
from global_config import *

def xsmm_asm_armv8_code(M, N, K, lda, ldb, ldc, pipeline_strategy_level, UNROLL_K, NR_MAIN, MRSA_FLAG, uniq_id):
    logger.debug(f"UNROLL_K: {UNROLL_K}")
    logger.debug(f"uniq_id: {uniq_id}")

    """Emit C code for gemm impl."""
    cc_code = f"""
#ifndef __SGEMM_KERNEL_H
#define __SGEMM_KERNEL_H
#endif

namespace laf {{
void small_gemm(const float *A, const float *B, float *C, int lda, int ldb, int ldc) {{
"""
    cc_code += laf_asm_code(M, N, K, lda, ldb, ldc, pipeline_strategy_level, UNROLL_K, NR_MAIN, MRSA_FLAG, with_bias = 0)
    cc_code += f"""
}}
void small_gemm_with_bias(const float *A, const float *B, float *C, int lda, int ldb, int ldc) {{
"""
    cc_code += laf_asm_code(M, N, K, lda, ldb, ldc, pipeline_strategy_level, UNROLL_K, NR_MAIN, MRSA_FLAG, with_bias = 1)
    cc_code += f"""
}}
}}

extern "C" int gemm_{M}x{N}x{K}_{lda}_{ldb}_{ldc}_xsmm_{uniq_id}(const float *A, const float *B, float *C, int lda, int ldb, int ldc){{
    laf::small_gemm(A, B, C, lda, ldb, ldc);
    return 0;
}}

extern "C" int gemm_{M}x{N}x{K}_{lda}_{ldb}_{ldc}_xsmm_with_bias_{uniq_id}(const float *A, const float *B, float *C, int lda, int ldb, int ldc){{
    laf::small_gemm_with_bias(A, B, C, lda, ldb, ldc);
    return 0;
}}
 
"""

    current_directory = os.path.dirname(os.path.abspath(__file__))
    generated_micro_kernel_path = os.path.join(current_directory, f'../../data/tune_output/build/generated_micro_kernel')
    generated_micro_kernel_file_path = os.path.join(generated_micro_kernel_path, f'c_file_asm_{M}_{K}_{N}_{lda}_{ldb}_{ldc}_{uniq_id}.cpp')
    with open(generated_micro_kernel_file_path, 'w') as f:
        f.write(cc_code)

    return cc_code