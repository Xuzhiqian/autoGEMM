from global_config import *
from unroll_loop_id import UNROLL_LOOP_ID
from micro_kernel_loop_asm import micro_kernel_loop_asm

def compile_time_for_loop_k_begin_func_asm(
    LINES, COLS,
    K, UNROLL_K,
    real_lines, real_cols,
    vector_id_array_A, VEC_REG_A_LEN,
    vector_id_array_B, VEC_REG_B_LEN,
    register_scroll_B,
    with_bias
):
    code_str = f""

    logger.debug("进入了K方向的初始化...")
    code_str += "\"\\n\" // 进入了K方向的初始化...\n"
    LOOP_K_BEGIN_FLAG = 1 # micro_kernel当中该参数的含义是K方向起始，可优化为fmul
    LOOP_K_END_FLAG = 0 # 显然不是K方向的末尾
    REG_BLOCK_TRANS_FLAG = 0
    FMA_CALCULATE_FLAG = 1
    STORE_C_FLAG = 0

    MAIN_K_LOOP_BEGIN = 0
    MAIN_K_LOOP_END, _ = UNROLL_LOOP_ID(K, UNROLL_K)

    tmp_LINES = real_lines
    cnt = 0
    while (tmp_LINES != 0):
        if (tmp_LINES % 2 != 0): # tmp_LINES is odd，如果tmp_LINES是偶数的话，这里将一直进不去，直到出现第一个奇数
            if cnt == 0: # first time
                code_str += f"    \"add     x10, x10, x6               \\n\" // A矩阵头指针加上x6(lda*FLOAT_BYTES)\n"
                code_str += f"    \"add     x13, x13, x9               \\n\" // C矩阵头指针加上x9(ldc*FLOAT_BYTES)\n"
            else: # other times
                code_str += f"    \"add     x10, x10, x6, lsl #{cnt}               \\n\" // A矩阵头指针加上x6*2^{cnt}倍\n"
                code_str += f"    \"add     x13, x13, x9, lsl #{cnt}               \\n\" // A矩阵头指针加上x6*2^{cnt}倍\n"
        tmp_LINES = tmp_LINES // 2 # 如果tmp_LINES 是偶数
        cnt += 1

    for LOOP_ID in range(MAIN_K_LOOP_BEGIN, MAIN_K_LOOP_END):
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
            with_bias
        )

    logger.debug("进入了K方向的初始化...完成")
    code_str += "\"\\n\" // 进入了K方向的初始化...完成\n"

    return code_str
