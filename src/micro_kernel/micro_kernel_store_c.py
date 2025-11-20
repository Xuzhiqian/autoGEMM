from global_config import *
from micro_kernel_common import get_vector_C_idx
from micro_kernel_common import get_x_C_idx
from micro_kernel_common import get_last_simd_col
from micro_kernel_common import get_simd_col
from micro_kernel_common import store_C_data
from micro_kernel_common import get_permuted_line

def micro_kernel_store_c(line, col,
                         UNROLL_NR,
                         vector_id_array_A, VEC_REG_A_LEN,
                         VEC_REG_B_LEN,
                         is_A_odd,
                         LINES, COLS,
                         real_lines, real_cols,
                         WITH_BIAS_FLAG):
    code_str = ""
    for j in range(UNROLL_NR): # 控制在列方向（N方向）的展开次数
        if line < real_lines:
            actual_line = line
            if is_A_odd and (not WITH_BIAS_FLAG):
                actual_line = get_permuted_line(line, real_lines, VEC_REG_A_LEN)
            vector_C_idx = get_vector_C_idx(actual_line, col, UNROLL_NR, j, COLS, VEC_REG_A_LEN, VEC_REG_B_LEN)
            last_simd_col = get_last_simd_col(col, UNROLL_NR, j)
            simd_col = get_simd_col(col, UNROLL_NR, j)
            x_C_idx = get_x_C_idx(actual_line)
            code_str += store_C_data(vector_C_idx, x_C_idx, simd_col, last_simd_col, real_cols)
    return code_str