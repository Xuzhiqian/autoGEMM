from unroll_loop_id import UNROLL_LOOP_ID
from micro_kernel_loop_asm import micro_kernel_loop_asm
from micro_kernel_common import prefetch_C_data

def compile_time_for_loop_k_end_main_loop_func_asm(
    LINES, COLS,
    K, UNROLL_K,
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
):
    code_str = ""

    _, REMAIN_K_LOOP_BEGIN = UNROLL_LOOP_ID(K, UNROLL_K)
    REMAIN_K_LOOP_END = UNROLL_K if K % UNROLL_K == 0 else K % UNROLL_K

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
    return code_str

def compile_time_for_loop_k_end_last_loop_func_asm(
    LINES, COLS,
    K, UNROLL_K,
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
):
    code_str = ""

    _, REMAIN_K_LOOP_BEGIN = UNROLL_LOOP_ID(K, UNROLL_K)
    REMAIN_K_LOOP_END = UNROLL_K if K % UNROLL_K == 0 else K % UNROLL_K

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

def compile_time_for_loop_k_end_last_loop_lt2_func_asm(
    LINES, COLS,
    real_lines, real_cols,
    vector_id_array_A, VEC_REG_A_LEN,
    vector_id_array_B, VEC_REG_B_LEN,
    register_scroll_B,
    LOOP_K_BEGIN_FLAG, LOOP_K_END_FLAG,
    REG_BLOCK_TRANS_FLAG,
    WITH_BIAS_FLAG
):
    code_str = ""
    LOOP_ID = -1
    LAST_K_ID = -1
    FMA_CALCULATE_FLAG = 0
    STORE_C_FLAG = 1
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

def compile_time_for_loop_k_end_last_loop_lt3_func_asm(
    LINES, COLS,
    real_lines, real_cols,
    vector_id_array_A, VEC_REG_A_LEN,
    vector_id_array_B, VEC_REG_B_LEN,
    register_scroll_B,
    LOOP_K_BEGIN_FLAG, LOOP_K_END_FLAG,
    REG_BLOCK_TRANS_FLAG,
    WITH_BIAS_FLAG
):
    code_str = ""
    LOOP_ID = -1
    LAST_K_ID = -1
    FMA_CALCULATE_FLAG = 0
    STORE_C_FLAG = 0
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

def compile_time_for_loop_k_end_main_loop_internal_func_asm(
    LINES, COLS,
    K, UNROLL_K,
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
):
    code_str = ""
    code_str += prefetch_C_data(real_lines)

    code_str += compile_time_for_loop_k_end_main_loop_func_asm(
        LINES, COLS,
        K, UNROLL_K,
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

    code_str += compile_time_for_loop_k_end_last_loop_func_asm(
        LINES, COLS,
        K, UNROLL_K,
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

def compile_time_for_loop_k_end_func_asm(
    LINES, COLS,
    K, UNROLL_K,
    real_lines, real_cols,
    vector_id_array_A, VEC_REG_A_LEN,
    vector_id_array_B, VEC_REG_B_LEN,
    register_scroll_B,
    pipeline_strategy_level
):
    LOOP_K_BEGIN_FLAG = 0 # pipeline_strategy_level < 3
    LOOP_K_END_FLAG = 1
    REG_BLOCK_TRANS_FLAG = 0
    FMA_CALCULATE_FLAG = 1
    STORE_C_FLAG = 1
    if pipeline_strategy_level < 2:
        STORE_C_FLAG = 0 # 通过lt2的额外的一次循环来单独处理store_c
    WITH_BIAS_FLAG = 0

    code_str = f""

    code_str += compile_time_for_loop_k_end_main_loop_internal_func_asm(
        LINES, COLS,
        K, UNROLL_K,
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

    if pipeline_strategy_level < 2:
        code_str += compile_time_for_loop_k_end_last_loop_lt2_func_asm(
            LINES, COLS,
            real_lines, real_cols,
            vector_id_array_A, VEC_REG_A_LEN,
            vector_id_array_B, VEC_REG_B_LEN,
            register_scroll_B,
            LOOP_K_BEGIN_FLAG, LOOP_K_END_FLAG,
            REG_BLOCK_TRANS_FLAG,
            WITH_BIAS_FLAG
        )

    return code_str