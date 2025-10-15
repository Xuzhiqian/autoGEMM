from unroll_loop_id import UNROLL_LOOP_ID
from micro_kernel_loop_asm import micro_kernel_loop_asm

def compile_time_for_loop_k_remain_func_asm(
    LINES, COLS,
    K, UNROLL_K,
    real_lines, real_cols,
    vector_id_array_A, VEC_REG_A_LEN,
    vector_id_array_B, VEC_REG_B_LEN,
    register_scroll_B
):
    LOOP_K_BEGIN_FLAG = 0
    LOOP_K_END_FLAG = 0
    REG_BLOCK_TRANS_FLAG = 0
    FMA_CALCULATE_FLAG = 1
    STORE_C_FLAG = 0
    WITH_BIAS_FLAG = 0

    REMAIN_K_LOOP_BEGIN, REMAIN_K_LOOP_END = UNROLL_LOOP_ID(K, UNROLL_K)

    code_str = f""

    # REMAIN_K_LOOP_BEGIN恒为1，只有当REMAIN_K_LOOP_END大于1等于UNROLL_K-SIMD_LANE时下面循环才会生效
    for LOOP_ID in range(REMAIN_K_LOOP_BEGIN, REMAIN_K_LOOP_END):
        LAST_K_ID = -1
        code_str += micro_kernel_loop_asm(
            LOOP_ID, LAST_K_ID,
            LINES, COLS,
            real_lines, real_cols,
            real_lines, real_cols,
            vector_id_array_A, VEC_REG_A_LEN,
            vector_id_array_B, VEC_REG_B_LEN,
            register_scroll_B,
            LOOP_K_BEGIN_FLAG, LOOP_K_END_FLAG,
            REG_BLOCK_TRANS_FLAG,
            FMA_CALCULATE_FLAG,
            STORE_C_FLAG,
            WITH_BIAS_FLAG
        )

    return code_str