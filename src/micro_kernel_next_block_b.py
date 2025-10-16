from global_config import *
from micro_kernel_common import get_simd_col
from micro_kernel_common import get_last_simd_col

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
    logger.debug("进入了B矩阵数据加载...")
    code_str += "\"\\n\" // 进入了B矩阵数据加载...\n"
    for j in range(UNROLL_NR):
        last_simd_col = get_last_simd_col(col, UNROLL_NR, j)
        simd_col = get_simd_col(col, UNROLL_NR, j)
        if (
            ((not is_last_k) or (is_last_k and COLS == VEC_REG_B_LEN)) and
            last_simd_col < real_cols
        ):
            if (
                LOOP_ID == LAST_K_ID - 1 and
                simd_col >= 2 * COLS - VEC_REG_B_LEN
            ):
                continue
            # B矩阵的数据加载和x寄存器偏移是交叉的
            vector_B_idx = vector_id_array_B[vector_scroll_B[simd_col]]
            x_B_idx = register_scroll_B[B_odd_flag]
            if SIMD == "NEON":
                code_str += f"    \"ldr     q{vector_B_idx}, [x{x_B_idx}, #{(ptr_B_POS) * SIMD_BYTES}]             \\n\" // 将x{x_B_idx} + #{(ptr_B_POS) * SIMD_BYTES} Bytes处的数据加载到q{vector_B_idx}当中\n"
            if SIMD == "SVE":
                code_str += f"    \"{LD1}     z{vector_B_idx}.{VEC_SIGN}, p0/z, [x{x_B_idx}, #{ptr_B_POS}, mul vl]             \\n\" // 将x{x_B_idx} + #{(ptr_B_POS) * SIMD_BYTES} Bytes处的数据加载到q{vector_B_idx}当中\n"
            # Get next B address
            if ptr_B_POS == COLS - 1: # last col
                ptr_B_POS = 0
                if SIMD == "NEON":
                    code_str += f"    \"add     x{x_B_idx}, x{x_B_idx}, {LDB}              \\n\" // 将x{x_B_idx}加上{LDB}后存入x{x_B_idx}\n"
                if SIMD == "SVE":
                    code_str += f"    \"add     x{x_B_idx}, x{x_B_idx}, %[ldb]              \\n\" // 将x{x_B_idx}加上%[ldb]后存入x{x_B_idx}\n"
                B_odd_flag ^= 1
            else:
                ptr_B_POS += 1
    logger.debug("进入了B矩阵数据加载...完成")
    code_str += "\"\\n\" // 进入了B矩阵数据加载...完成\n"
    return code_str, ptr_B_POS, B_odd_flag