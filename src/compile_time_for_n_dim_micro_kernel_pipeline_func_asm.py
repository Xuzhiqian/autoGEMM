from global_config import *
from compile_time_for_loop_k_end_func_asm import compile_time_for_loop_k_end_func_asm
from compile_time_for_loop_k_end_func_asm import compile_time_for_loop_k_end_last_loop_lt3_func_asm
from compile_time_for_loop_k_end_func_asm import compile_time_for_loop_k_end_main_loop_internal_func_asm

def compile_time_for_n_dim_micro_kernel_pipeline_lt3_func_asm(
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
    logger.debug(f"进入了loop_k_end_func...")
    code_str += "\"\\n\" // 进入了loop_k_end_func...\n"
    code_str += compile_time_for_loop_k_end_func_asm(
        LINES, COLS,
        K, UNROLL_K,
        real_lines, real_cols,
        vector_id_array_A, VEC_REG_A_LEN,
        vector_id_array_B, VEC_REG_B_LEN,
        register_scroll_B,
        pipeline_strategy_level
    )
    logger.debug(f"进入了loop_k_end_func...完成")
    code_str += "\"\\n\" // 进入了loop_k_end_func...完成\n"
    LOOP_K_BEGIN_FLAG = 0
    LOOP_K_END_FLAG = 0
    REG_BLOCK_TRANS_FLAG = 2
    WITH_BIAS_FLAG = with_bias
    logger.debug(f"进入了loop_k_end_last_loop_lt3_func...")
    code_str += "\"\\n\" // 进入了loop_k_end_last_loop_lt3_func...\n"
    code_str += compile_time_for_loop_k_end_last_loop_lt3_func_asm(
        LINES, COLS,
        real_lines, real_cols,
        next_lines, next_cols,
        vector_id_array_A, VEC_REG_A_LEN,
        vector_id_array_B, VEC_REG_B_LEN,
        register_scroll_B,
        LOOP_K_BEGIN_FLAG, LOOP_K_END_FLAG,
        REG_BLOCK_TRANS_FLAG,
        WITH_BIAS_FLAG
    )
    logger.debug(f"进入了loop_k_end_last_loop_lt3_func...完成")
    code_str += "\"\\n\" // 进入了loop_k_end_last_loop_lt3_func...完成\n"
    return code_str

def compile_time_for_n_dim_micro_kernel_pipeline_eq3_func_asm(
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
    REG_BLOCK_TRANS_FLAG = 2
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

def compile_time_for_n_dim_micro_kernel_pipeline_func_asm(
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
        logger.debug(f"进入了pipeline<3的NR_LOOPS循环...")
        code_str += "\"\\n\" // 进入了pipeline<3的NR_LOOPS循环...\n"
        code_str += compile_time_for_n_dim_micro_kernel_pipeline_lt3_func_asm(
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
        logger.debug(f"进入了pipeline<3的NR_LOOPS循环...完成")
        code_str += "\"\\n\" // 进入了pipeline<3的NR_LOOPS循环...完成\n"
    else:
        logger.debug(f"进入了pipeline=3的NR_LOOPS循环...")
        code_str += "\"\\n\" // 进入了pipeline=3的NR_LOOPS循环...\n"
        code_str += compile_time_for_n_dim_micro_kernel_pipeline_eq3_func_asm(
            LINES, COLS,
            K, UNROLL_K,
            real_lines, real_cols,
            next_lines, next_cols,
            vector_id_array_A, VEC_REG_A_LEN,
            vector_id_array_B, VEC_REG_B_LEN,
            register_scroll_B,
            with_bias
        )
        logger.debug(f"进入了pipeline=3的NR_LOOPS循环...完成")
        code_str += "\"\\n\" // 进入了pipeline=3的NR_LOOPS循环...完成\n"

    return code_str