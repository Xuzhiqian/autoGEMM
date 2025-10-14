from unroll_loop_id import UNROLL_LOOP_ID
from micro_kernel_loop_asm import micro_kernel_loop_asm

def compile_time_for_loop_k_main_body_func_asm(LINES, COLS, K, UNROLL_K, real_lines, real_cols, vector_id_array_A, VEC_REG_A_LEN, vector_id_array_B, VEC_REG_B_LEN, register_scroll_B):
    LOOP_K_BEGIN_FLAG = 0
    LOOP_K_END_FLAG = 0
    REG_BLOCK_TRANS_FLAG = 0
    FMA_CALCULATE_FLAG = 1
    STORE_C_FLAG = 0
    WITH_BIAS_FLAG = 0

    MAIN_K_LOOP_BEGIN, _ = UNROLL_LOOP_ID(K, UNROLL_K)
    MAIN_K_LOOP_END = UNROLL_K + MAIN_K_LOOP_BEGIN

    code_str = f""

    for LOOP_ID in range(MAIN_K_LOOP_BEGIN, MAIN_K_LOOP_END):
      code_str += micro_kernel_loop_asm(LOOP_ID, -1, LINES, COLS, real_lines, real_cols, real_lines, real_cols, vector_id_array_A, VEC_REG_A_LEN, vector_id_array_B, VEC_REG_B_LEN, register_scroll_B, LOOP_K_BEGIN_FLAG, LOOP_K_END_FLAG, REG_BLOCK_TRANS_FLAG, FMA_CALCULATE_FLAG, STORE_C_FLAG, WITH_BIAS_FLAG)

    return code_str