from global_config import *

def get_vector_B_idx(col,
                     UNROLL_NR,
                     j,
                     vector_id_array_B,
                     vector_scroll_B):
    return vector_id_array_B[vector_scroll_B[col * UNROLL_NR + j]]

def get_vector_C_idx(line, col,
                     UNROLL_NR,
                     j,
                     COLS):
    vector_C_idx = line * COLS + col * UNROLL_NR + j
    return vector_C_idx

def get_simd_col(col,
                 UNROLL_NR,
                 j):
    return col * UNROLL_NR + j

def get_last_simd_col(col,
                      UNROLL_NR,
                      j):
    last_simd_col = SIMD_LANE * (col * UNROLL_NR + j)
    return last_simd_col

def prefetch_C_data(real_lines):
    code_str = ""
    for line in range(real_lines):
        x_C_idx = RESERVED_REG_NUM + line
        code_str += f"    \"prfm    PSTL1KEEP, [x{x_C_idx}, #64]              \\n\" // 从x{x_C_idx}预取C矩阵数据\n"
    return code_str