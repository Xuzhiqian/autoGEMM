from global_config import RESERVED_REG_NUM
from unroll_loop_id import UNROLL_LOOP_ID
from micro_kernel_loop_asm import micro_kernel_loop_asm
from compile_time_for_loop_k_end_func_asm import compile_time_for_loop_k_end_func_asm

def compile_time_for_m_dim_micro_kernel_pipeline_func_asm(
    LINES, COLS,
    K, UNROLL_K,
    real_lines, real_cols,
    next_lines, next_cols,
    vector_id_array_A, VEC_REG_A_LEN,
    vector_id_array_B, VEC_REG_B_LEN,
    register_scroll_B,
    with_bias,
    pipeline_strategy_level
):
    LOOP_K_BEGIN_FLAG = 0
    LOOP_K_END_FLAG = 0
    REG_BLOCK_TRANS_FLAG = 1
    FMA_CALCULATE_FLAG = 1
    STORE_C_FLAG = 1
    WITH_BIAS_FLAG = with_bias

    _, REMAIN_K_LOOP_BEGIN = UNROLL_LOOP_ID(K, UNROLL_K)
    REMAIN_K_LOOP_END = UNROLL_K if K % UNROLL_K == 0 else K % UNROLL_K
    
    code_str = f""

    if pipeline_strategy_level < 3: 
        code_str += compile_time_for_loop_k_end_func_asm(
            LINES, COLS,
            K, UNROLL_K,
            real_lines, real_cols,
            vector_id_array_A, VEC_REG_A_LEN,
            vector_id_array_B, VEC_REG_B_LEN,
            register_scroll_B,
            pipeline_strategy_level
        )
        LOOP_ID = -1
        LAST_K_ID = -1
        FMA_CALCULATE_FLAG = 0
        STORE_C_FLAG = 0
        code_str += micro_kernel_loop_asm(
            LOOP_ID, LAST_K_ID,
            LINES, COLS,
            real_lines, real_cols,
            next_lines, next_cols,
            vector_id_array_A, VEC_REG_A_LEN,
            vector_id_array_B, VEC_REG_B_LEN,
            register_scroll_B,
            LOOP_K_BEGIN_FLAG, LOOP_K_END_FLAG,
            REG_BLOCK_TRANS_FLAG,
            FMA_CALCULATE_FLAG,
            STORE_C_FLAG,
            WITH_BIAS_FLAG
        )
    else:
        for line in range(real_lines):
            code_str += f"    \"prfm    PSTL1KEEP, [x{RESERVED_REG_NUM+line}, #64]              \\n\"\n"

        for LOOP_ID in range(REMAIN_K_LOOP_BEGIN, REMAIN_K_LOOP_END):
            LAST_K_ID = REMAIN_K_LOOP_END - 1
            code_str += micro_kernel_loop_asm(
                LOOP_ID, LAST_K_ID,
                LINES, COLS,
                real_lines, real_cols,
                next_lines, next_cols,
                vector_id_array_A, VEC_REG_A_LEN,
                vector_id_array_B, VEC_REG_B_LEN,
                register_scroll_B,
                LOOP_K_BEGIN_FLAG, LOOP_K_END_FLAG,
                REG_BLOCK_TRANS_FLAG,
                FMA_CALCULATE_FLAG,
                STORE_C_FLAG,
                WITH_BIAS_FLAG
            )

        # When K module UNROLL_K remainder 1, no calculation, direct store
        if REMAIN_K_LOOP_BEGIN == REMAIN_K_LOOP_END:
            LOOP_ID = -1
            LAST_K_ID = -1
            FMA_CALCULATE_FLAG = 0
            code_str += micro_kernel_loop_asm(
                LOOP_ID, LAST_K_ID,
                LINES, COLS,
                real_lines, real_cols,
                next_lines, next_cols,
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