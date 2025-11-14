from global_config import *
from micro_kernel_common import get_simd_col
from micro_kernel_common import get_last_simd_col
from micro_kernel_common import get_vector_B_idx
from micro_kernel_common import load_B_data_and_offset

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
            vector_B_idx = get_vector_B_idx(simd_col, vector_id_array_B, vector_scroll_B)
            x_B_idx = register_scroll_B[B_odd_flag]
            code_str_b, ptr_B_POS, B_odd_flag = load_B_data_and_offset(vector_B_idx, x_B_idx, ptr_B_POS, B_odd_flag, COLS)
            code_str += code_str_b
    logger.debug("进入了B矩阵数据加载...完成")
    code_str += "\"\\n\" // 进入了B矩阵数据加载...完成\n"
    return code_str, ptr_B_POS, B_odd_flag