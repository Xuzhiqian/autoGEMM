from global_config import *

def micro_kernel_next_block_a_get_addr(line, col,
                                       LINES, COLS,
                                       next_lines, next_cols):
    code_str = ""
    is_first_line = (line == 0)
    is_first_col = (col == 0)
    if is_first_line and is_first_col: # 第0行第0列
        logger.debug("进入了A矩阵x寄存器初始化...")
        code_str += "\"\\n\" // 进入了A矩阵x寄存器初始化...\n"
        logger.debug(f"next_lines = {next_lines}")
        code_str += f"\"\\n\" // next_lines = {next_lines}\n"
        for j in range(next_lines):
            x_A_idx = RESERVED_REG_NUM + LINES + j
            if j == 0:
                code_str += f"    \"mov     x{x_A_idx}, {A_Head}    \\n\" // 将{A_Head}(A矩阵头指针)存入x{x_A_idx}\n"
            elif j == 1:
                if SIMD == "NEON":
                    code_str += f"    \"add     x{x_A_idx}, {A_Head}, {LDA}    \\n\" // 将{A_Head}加上{LDA}后存入x{x_A_idx}\n"
                if SIMD == "SVE":
                    code_str += f"    \"add     x{x_A_idx}, {A_Head}, %[lda]    \\n\" // 将{A_Head}加上%[lda]后存入x{x_A_idx}\n"
            else:
                if SIMD == "NEON":
                    code_str += f"    \"add     x{x_A_idx}, x{x_A_idx - 2}, {LDA}, lsl #1    \\n\"// 将x{x_A_idx - 2}加上2倍的{LDA}后存入x{x_A_idx}\n"
                if SIMD == "SVE":
                    code_str += f"    \"add     x{x_A_idx}, x{x_A_idx - 2}, %[lda], lsl #1    \\n\"// 将x{x_A_idx - 2}加上2倍的%[lda]后存入x{x_A_idx}\n"
        logger.debug("进入了A矩阵x寄存器初始化...完成")
        code_str += "\"\\n\" // 进入了A矩阵x寄存器初始化...完成\n"
    return code_str

def micro_kernel_next_block_a_load_data(line, col,
                                        UNROLL_NR,
                                        VEC_REG_A_LEN,
                                        vector_scroll_A,
                                        A_odd_flag,
                                        mod_simd_lane_loop_id,
                                        is_last_k,
                                        LOOP_ID, LAST_K_ID,
                                        LINES, COLS,
                                        real_lines, real_cols,
                                        next_lines, next_cols,
                                        REG_BLOCK_TRANS_FLAG,
                                        WITH_BIAS_FLAG):
    is_A_even = (A_odd_flag == 0)
    code_str = ""
    if (REG_BLOCK_TRANS_FLAG == 0 and ( # unknown constraints
        (LAST_K_ID == -1 or LOOP_ID < (LAST_K_ID - LAST_K_ID % 4))
    )):
        logger.debug("进入了A矩阵数据加载...")
        code_str += "\"\\n\" // 进入了A矩阵数据加载...\n"
        is_first_line = (line == 0)
        is_first_col = (col == 0)
        if is_first_line and is_first_col: # 第0行第0列
            for real_line in range(real_lines):
                if ( # unknown constraints
                    (UNROLL_LANE == 1 or mod_simd_lane_loop_id == real_line % (UNROLL_LANE - 1)) and
                    (real_line >= real_lines - VEC_REG_A_LEN % real_lines or 2 * real_lines <= VEC_REG_A_LEN)
                ):
                    actual_line = (real_line + VEC_REG_A_LEN % real_lines) % real_lines if is_A_even else real_line
                    x_A_idx = RESERVED_REG_NUM + LINES + actual_line
                    vector_A_idx = vector_scroll_A[A_odd_flag^1][actual_line]
                    if SIMD == "NEON":
                        code_str += f"    \"ldr     q{vector_A_idx}, [x{x_A_idx}], #{SIMD_BYTES}    \\n\"// 将x{x_A_idx}处的数据加载到q{vector_A_idx}中，并使x{x_A_idx}偏移SIMD的长度\n"
                    if SIMD == "SVE":
                        code_str += f"    \"{LD1R}     z{vector_A_idx}.{VEC_SIGN}, p0/z, [x{x_A_idx}]    \\n\"// 将x{x_A_idx}处的数据广播到q{vector_A_idx}中\n"
                        code_str += f"    \"add     x{x_A_idx}, x{x_A_idx}, #{UNROLL_LANE * FLOAT_BYTES}    \\n\"// 使x{x_A_idx}偏移FLOAT_BYTES的长度\n"
        if ( # unknown constraints
            mod_simd_lane_loop_id == (UNROLL_LANE - 1) and
            line < real_lines and
            col == COLS // UNROLL_NR - 1 and
            2 * real_lines > VEC_REG_A_LEN and
            line < real_lines - VEC_REG_A_LEN % real_lines
        ):
            actual_line = (line + VEC_REG_A_LEN % real_lines) % real_lines if is_A_even else line
            x_A_idx = RESERVED_REG_NUM + LINES + actual_line
            vector_A_idx = vector_scroll_A[A_odd_flag^1][actual_line]
            if SIMD == "NEON":
                code_str += f"    \"ldr     q{vector_A_idx}, [x{x_A_idx}], #{SIMD_BYTES}    \\n\" // 将x{x_A_idx}处的数据加载到q{vector_A_idx}中，并使x{x_A_idx}偏移SIMD的长度\n"
            if SIMD == "SVE":
                code_str += f"    \"{LD1R}     z{vector_A_idx}.{VEC_SIGN}, p0/z, [x{x_A_idx}]    \\n\"// 将x{x_A_idx}处的数据广播到q{vector_A_idx}中\n"
                code_str += f"    \"add     x{x_A_idx}, x{x_A_idx}, #{UNROLL_LANE * FLOAT_BYTES}    \\n\"// 使x{x_A_idx}偏移FLOAT_BYTES的长度\n"
                
        logger.debug("进入了A矩阵数据加载...完成")
        code_str += "\"\\n\" // 进入了A矩阵数据加载...完成\n"
        return code_str

    logger.debug("进入了A矩阵数据加载...")
    code_str += "\"\\n\" // 进入了A矩阵数据加载...\n"
    if ( # unknown constraints
        is_last_k and
        line < next_lines and
        col == COLS // UNROLL_NR - 1
    ):
        if ((not WITH_BIAS_FLAG) or ( # unknown constraints
                is_A_even or
                line >= real_lines - VEC_REG_A_LEN % real_lines
            )
        ):
            x_A_idx = RESERVED_REG_NUM + LINES + line
            vector_A_idx = vector_scroll_A[0][line]
            if SIMD == "NEON":
                code_str += f"    \"ldr     q{vector_A_idx}, [x{x_A_idx}], #{SIMD_BYTES}    \\n\" // 将x{x_A_idx}处的数据加载到q{vector_A_idx}当中，并使x{x_A_idx}往后偏移SIMD的长度\n"
            if SIMD == "SVE":
                code_str += f"    \"{LD1R}     z{vector_A_idx}.{VEC_SIGN}, p0/z, [x{x_A_idx}]    \\n\"// 将x{x_A_idx}处的数据广播到q{vector_A_idx}中\n"
                code_str += f"    \"add     x{x_A_idx}, x{x_A_idx}, #{UNROLL_LANE * FLOAT_BYTES}    \\n\"// 使x{x_A_idx}偏移FLOAT_BYTES的长度\n"
    logger.debug("进入了A矩阵数据加载...完成")
    code_str += "\"\\n\" // 进入了A矩阵数据加载...完成\n"
    return code_str

def micro_kernel_next_block_a(line, col,
                              UNROLL_NR,
                              VEC_REG_A_LEN,
                              vector_scroll_A,
                              A_odd_flag,
                              mod_simd_lane_loop_id,
                              is_last_k,
                              LOOP_ID, LAST_K_ID,
                              LINES, COLS,
                              real_lines, real_cols,
                              next_lines, next_cols,
                              REG_BLOCK_TRANS_FLAG,
                              WITH_BIAS_FLAG):
    code_str = ""
    if REG_BLOCK_TRANS_FLAG and is_last_k:
        code_str += micro_kernel_next_block_a_get_addr(line, col,
                                                       LINES, COLS,
                                                       next_lines, next_cols)

    code_str += micro_kernel_next_block_a_load_data(line, col,
                                                    UNROLL_NR,
                                                    VEC_REG_A_LEN,
                                                    vector_scroll_A,
                                                    A_odd_flag,
                                                    mod_simd_lane_loop_id,
                                                    is_last_k,
                                                    LOOP_ID, LAST_K_ID,
                                                    LINES, COLS,
                                                    real_lines, real_cols,
                                                    next_lines, next_cols,
                                                    REG_BLOCK_TRANS_FLAG,
                                                    WITH_BIAS_FLAG)

    return code_str