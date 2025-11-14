from global_config import *
from micro_kernel_common import get_vector_A_idx
from micro_kernel_common import load_A_data_and_offset

def micro_kernel_block_a_extra(is_A_odd,
                               VEC_REG_A_LEN,
                               vector_scroll_A,
                               LINES, COLS,
                               real_lines, real_cols,
                               next_lines, next_cols):
    code_str = ""
    for line in range(next_lines):
        if is_A_odd and line < real_lines - VEC_REG_A_LEN % real_lines:
            vector_A_idx = get_vector_A_idx(line, 0, vector_scroll_A)
            x_A_idx = RESERVED_REG_NUM + LINES + line
            code_str += load_A_data_and_offset(vector_A_idx, x_A_idx)
    return code_str