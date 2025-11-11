from global_config import *
from micro_kernel_loop_asm import micro_kernel_loop_asm

def compile_time_for_init_func_asm(
    LINES, COLS,
    real_lines, real_cols,
    vector_id_array_A, VEC_REG_A_LEN,
    vector_id_array_B, VEC_REG_B_LEN,
    register_scroll_B,
    with_bias
):
    code_str = ""

    logger.debug("进入了N方向的初始化...")
    code_str += "\"\\n\" // 进入了N方向的初始化...\n"
    LOOP_ID = -1
    LAST_K_ID = -1
    LOOP_K_BEGIN_FLAG = 0
    LOOP_K_END_FLAG = 0
    REG_BLOCK_TRANS_FLAG = 1
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
        with_bias
    )

    logger.debug("进入了N方向的初始化...完成")
    code_str += "\"\\n\" // 进入了N方向的初始化...完成\n"

    return code_str