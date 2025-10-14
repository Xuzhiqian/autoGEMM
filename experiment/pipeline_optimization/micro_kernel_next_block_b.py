from global_config import *

def micro_kernel_next_block_b(line, col,
                              UNROLL_NR,
                              vector_id_array_B, VEC_REG_B_LEN,
                              vector_scroll_B,
                              register_scroll_B,
                              B_odd_flag,
                              ptr_B_POS,
                              is_last_k,
                              LOOP_ID, LAST_K_ID,
                              LINES, COLS,
                              real_lines, real_cols):
    code_str = ""
    if (line == LINES - 1):
        logger.debug("进入了B矩阵数据加载...")
        code_str += "\"\\n\" // 进入了B矩阵数据加载...\n"
        for j in range(UNROLL_NR):
            unkown_param_1 = UNROLL_NR * col + j
            if (
                ((not is_last_k) or (is_last_k and COLS == VEC_REG_B_LEN)) and
                SIMD_LANE * (unkown_param_1) < real_cols
            ):
                if (
                    LOOP_ID == LAST_K_ID - 1 and
                    unkown_param_1 >= 2 * COLS - VEC_REG_B_LEN
                ):
                    continue
                # B矩阵的数据加载和x寄存器偏移是交叉的
                code_str += f"    \"ldr     q{vector_id_array_B[vector_scroll_B[unkown_param_1]]}, [x{register_scroll_B[B_odd_flag]}, #{(ptr_B_POS)*16}]             \\n\" // 将x{register_scroll_B[B_odd_flag]} + #{(ptr_B_POS)*16} Bytes处的数据加载到q{vector_id_array_B[vector_scroll_B[unkown_param_1]]}当中\n"
                # Get next B address
                if ptr_B_POS == COLS - 1:
                    ptr_B_POS = 0
                    code_str += f"    \"add     x{register_scroll_B[B_odd_flag]}, x{register_scroll_B[B_odd_flag]}, x8              \\n\" // 将x{register_scroll_B[B_odd_flag]}加上x8后存入x{register_scroll_B[B_odd_flag]}\n"
                    B_odd_flag ^= 1
                else:
                    ptr_B_POS += 1
        logger.debug("进入了B矩阵数据加载...完成")
        code_str += "\"\\n\" // 进入了B矩阵数据加载...完成\n"
    return code_str, ptr_B_POS