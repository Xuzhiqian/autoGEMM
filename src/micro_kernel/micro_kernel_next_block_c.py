from global_config import *
from micro_kernel_common import get_vector_C_idx
from micro_kernel_common import get_last_simd_col
from micro_kernel_common import get_simd_col
from micro_kernel_common import load_C_data

def micro_kernel_next_block_c_get_addr(line, col,
                                       UNROLL_NR,
                                       LINES, COLS,
                                       next_lines, next_cols,
                                       WITH_BIAS_FLAG):
    code_str = ""
    if not WITH_BIAS_FLAG: # 没有BIAS的话是一次性在最后一行最后一列将所有的C寄存器进行偏移
        is_last_line = (line == LINES - 1)
        is_last_col = (col == COLS // UNROLL_NR - 1)
        if is_last_line and is_last_col:
            logger.debug("进入了C矩阵x寄存器初始化...")
            code_str += "\"\\n\" // 进入了C矩阵x寄存器初始化...\n"
            for j in range(next_lines):
                x_C_idx = RESERVED_REG_NUM + j
                if j == 0:
                    code_str += f"    \"mov     x{x_C_idx}, {C_Head}    \\n\" // 将{C_Head}(C矩阵头指针)存入x{x_C_idx}\n"
                elif j == 1:
                    if SIMD == "NEON":
                        code_str += f"    \"add     x{x_C_idx}, {C_Head}, {LDC}     \\n\" // 将{C_Head}加上{LDC}后存入x{x_C_idx}\n"
                    if SIMD == "SVE":
                        code_str += f"    \"add     x{x_C_idx}, {C_Head}, %[ldc]     \\n\" // 将{C_Head}加上%[ldc]后存入x{x_C_idx}\n"
                else:
                    if SIMD == "NEON":
                        code_str += f"    \"add     x{x_C_idx}, x{x_C_idx - 2}, {LDC}, lsl #1    \\n\" // 将x{x_C_idx - 2}加上2倍的{LDC}后存入x{x_C_idx}\n"
                    if SIMD == "SVE":
                        code_str += f"    \"add     x{x_C_idx}, x{x_C_idx - 2}, %[ldc], lsl #1    \\n\" // 将x{x_C_idx - 2}加上2倍的%[ldc]后存入x{x_C_idx}\n"
    else: # 有BIAS的话是每行都进行C寄存器的偏移
        logger.debug("进入了C矩阵x寄存器初始化...")
        code_str += "\"\\n\" // 进入了C矩阵x寄存器初始化...\n"
        if line < next_lines:
            x_C_idx = RESERVED_REG_NUM + line
            if line == 0:
                code_str += f"    \"mov     x{x_C_idx}, {C_Head}    \\n\"\n"
            elif line == 1:
                if SIMD == "NEON":
                    code_str += f"    \"add     x{x_C_idx}, {C_Head}, {LDC}     \\n\"\n"
                if SIMD == "SVE":
                    code_str += f"    \"add     x{x_C_idx}, {C_Head}, %[ldc]     \\n\"\n"
            else:
                if SIMD == "NEON":
                    code_str += f"    \"add     x{x_C_idx}, x{x_C_idx - 2}, {LDC}, lsl #1    \\n\"\n"
                if SIMD == "SVE":
                    code_str += f"    \"add     x{x_C_idx}, x{x_C_idx - 2}, %[ldc], lsl #1    \\n\"\n"

    logger.debug("进入了C矩阵x寄存器初始化...完成")
    code_str += "\"\\n\" // 进入了C矩阵x寄存器初始化...完成\n"
    return code_str

def micro_kernel_next_block_c_load_data(line, col,
                                        UNROLL_NR,
                                        LINES, COLS,
                                        VEC_REG_A_LEN,
                                        VEC_REG_B_LEN,
                                        next_lines, next_cols,
                                        WITH_BIAS_FLAG):
    code_str = ""
    if not WITH_BIAS_FLAG:
        return code_str

    logger.debug("进入了C矩阵数据加载...")
    code_str += "\"\\n\" // 进入了C矩阵数据加载...\n"
    for j in range(UNROLL_NR):
        last_simd_col = get_last_simd_col(col, UNROLL_NR, j)
        simd_col = get_simd_col(col, UNROLL_NR, j)
        if line < next_lines and last_simd_col < next_cols:
            vector_C_idx = get_vector_C_idx(line, col, UNROLL_NR, j, COLS, VEC_REG_A_LEN, VEC_REG_B_LEN)
            x_C_idx = RESERVED_REG_NUM + line
            code_str += load_C_data(vector_C_idx, x_C_idx, simd_col)
    logger.debug("进入了C矩阵数据加载...完成")
    code_str += "\"\\n\" // 进入了C矩阵数据加载...完成\n"
    return code_str

def micro_kernel_next_block_c(line, col,
                              UNROLL_NR,
                              LINES, COLS,
                              VEC_REG_A_LEN,
                              VEC_REG_B_LEN,
                              next_lines, next_cols,
                              WITH_BIAS_FLAG):

    code_str = ""
    logger.debug("进入了micro_kernel_next_block_c_get_addr...")
    code_str += "\"\\n\" // 进入了micro_kernel_next_block_c_get_addr...\n"
    code_str += micro_kernel_next_block_c_get_addr(line, col,
                                                   UNROLL_NR,
                                                   LINES, COLS,
                                                   next_lines, next_cols,
                                                   WITH_BIAS_FLAG)
    logger.debug("进入了micro_kernel_next_block_c_get_addr...完成")
    code_str += "\"\\n\" // 进入了micro_kernel_next_block_c_get_addr...完成\n"

    logger.debug("进入了micro_kernel_next_block_c_load_data...")
    code_str += "\"\\n\" // 进入了micro_kernel_next_block_c_load_data...\n"
    code_str += micro_kernel_next_block_c_load_data(line, col,
                                                    UNROLL_NR,
                                                    LINES, COLS,
                                                    VEC_REG_A_LEN,
                                                    VEC_REG_B_LEN,
                                                    next_lines, next_cols,
                                                    WITH_BIAS_FLAG)
    logger.debug("进入了micro_kernel_next_block_c_load_data...完成")
    code_str += "\"\\n\" // 进入了micro_kernel_next_block_c_load_data...完成\n"

    return code_str