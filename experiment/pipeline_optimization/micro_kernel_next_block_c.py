from global_config import *
from micro_kernel_common import get_vector_C_idx
from micro_kernel_common import get_last_simd_col

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
                if j == 0:
                    code_str += f"    \"mov     x{RESERVED_REG_NUM}, x13    \\n\" // 将x13(C矩阵头指针)存入x{RESERVED_REG_NUM}\n"
                elif j == 1:
                    code_str += f"    \"add     x{RESERVED_REG_NUM + 1}, x13, x9     \\n\" // 将x13加上x9后存入x{RESERVED_REG_NUM + 1}\n"
                else:
                    code_str += f"    \"add     x{RESERVED_REG_NUM + j}, x{RESERVED_REG_NUM + j - 2}, x9, lsl #1    \\n\" // 将x{RESERVED_REG_NUM + j-2}加上2倍的x9后存入x{RESERVED_REG_NUM + j}\n"
            logger.debug("进入了C矩阵x寄存器初始化...完成")
            code_str += "\"\\n\" // 进入了C矩阵x寄存器初始化...完成\n"
    else: # 有BIAS的话是每行都进行C寄存器的偏移
        logger.debug("进入了C矩阵x寄存器初始化...")
        code_str += "\"\\n\" // 进入了C矩阵x寄存器初始化...\n"
        if line < next_lines:
            if line == 0:
                code_str += f"    \"mov     x{RESERVED_REG_NUM}, x13    \\n\"\n"
            elif line == 1:
                code_str += f"    \"add     x{RESERVED_REG_NUM + 1}, x13, x9     \\n\"\n"
            else:
                code_str += f"    \"add     x{RESERVED_REG_NUM + line}, x{RESERVED_REG_NUM + line - 2}, x9, lsl #1    \\n\"\n"
        logger.debug("进入了C矩阵x寄存器初始化...完成")
        code_str += "\"\\n\" // 进入了C矩阵x寄存器初始化...完成\n"

    return code_str

def micro_kernel_next_block_c_load_data(line, col,
                                        UNROLL_NR,
                                        LINES, COLS,
                                        next_lines, next_cols,
                                        WITH_BIAS_FLAG):
    code_str = ""
    if not WITH_BIAS_FLAG:
        return code_str

    logger.debug("进入了C矩阵数据加载...")
    code_str += "\"\\n\" // 进入了C矩阵数据加载...\n"
    for j in range(UNROLL_NR):
        last_simd_col = get_last_simd_col(col, UNROLL_NR, j)
        if line < next_lines and last_simd_col < next_cols:
            vector_C_idx = get_vector_C_idx(line, col, UNROLL_NR, j, COLS)
            code_str += f"    \"ldr     q{vector_C_idx}, [x{RESERVED_REG_NUM + line}, #{(col * UNROLL_NR + j) * 16}]           \\n\"\n"
    logger.debug("进入了C矩阵数据加载...完成")
    code_str += "\"\\n\" // 进入了C矩阵数据加载...完成\n"
    return code_str

def micro_kernel_next_block_c(line, col,
                              UNROLL_NR,
                              is_last_k,
                              LINES, COLS,
                              next_lines, next_cols,
                              REG_BLOCK_TRANS_FLAG,
                              WITH_BIAS_FLAG):

    code_str = ""
    if REG_BLOCK_TRANS_FLAG and is_last_k: # 只有启用REG_BLOCK_TRANS_FLAG及最后一个K时才需要进行C矩阵的x寄存器的偏移
        code_str += micro_kernel_next_block_c_get_addr(line, col,
                                                    UNROLL_NR,
                                                    LINES, COLS,
                                                    next_lines, next_cols,
                                                    WITH_BIAS_FLAG)

        code_str += micro_kernel_next_block_c_load_data(line, col,
                                                        UNROLL_NR,
                                                        LINES, COLS,
                                                        next_lines, next_cols,
                                                        WITH_BIAS_FLAG)

    return code_str