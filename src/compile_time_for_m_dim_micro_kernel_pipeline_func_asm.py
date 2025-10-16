from compile_time_for_loop_k_end_func_asm import compile_time_for_loop_k_end_func_asm
from compile_time_for_loop_k_end_func_asm import compile_time_for_loop_k_end_last_loop_lt3_func_asm
from compile_time_for_loop_k_end_func_asm import compile_time_for_loop_k_end_main_loop_internal_func_asm

def compile_time_for_m_dim_micro_kernel_pipeline_lt3_func_asm(
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
    code_str = ""
    code_str += compile_time_for_loop_k_end_func_asm(
        LINES, COLS,
        K, UNROLL_K,
        real_lines, real_cols,
        vector_id_array_A, VEC_REG_A_LEN,
        vector_id_array_B, VEC_REG_B_LEN,
        register_scroll_B,
        pipeline_strategy_level
    )
    LOOP_K_BEGIN_FLAG = 0
    LOOP_K_END_FLAG = 0
    REG_BLOCK_TRANS_FLAG = 1
    WITH_BIAS_FLAG = with_bias
    code_str += compile_time_for_loop_k_end_last_loop_lt3_func_asm(
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

def compile_time_for_m_dim_micro_kernel_pipeline_eq3_func_asm(
    LINES, COLS,
    K, UNROLL_K,
    real_lines, real_cols,
    next_lines, next_cols,
    vector_id_array_A, VEC_REG_A_LEN,
    vector_id_array_B, VEC_REG_B_LEN,
    register_scroll_B,
    with_bias
):
    code_str = ""

    LOOP_K_BEGIN_FLAG = 0
    LOOP_K_END_FLAG = 0
    REG_BLOCK_TRANS_FLAG = 1
    FMA_CALCULATE_FLAG = 1
    STORE_C_FLAG = 1
    WITH_BIAS_FLAG = with_bias

    code_str += compile_time_for_loop_k_end_main_loop_internal_func_asm(
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
    code_str = f""

    if pipeline_strategy_level < 3:
        code_str += compile_time_for_m_dim_micro_kernel_pipeline_lt3_func_asm(
            LINES, COLS,
            K, UNROLL_K,
            real_lines, real_cols,
            next_lines, next_cols,
            vector_id_array_A, VEC_REG_A_LEN,
            vector_id_array_B, VEC_REG_B_LEN,
            register_scroll_B,
            with_bias,
            pipeline_strategy_level
        )
    else:
        code_str += compile_time_for_m_dim_micro_kernel_pipeline_eq3_func_asm(
            LINES, COLS,
            K, UNROLL_K,
            real_lines, real_cols,
            next_lines, next_cols,
            vector_id_array_A, VEC_REG_A_LEN,
            vector_id_array_B, VEC_REG_B_LEN,
            register_scroll_B,
            with_bias
        )

    return code_str