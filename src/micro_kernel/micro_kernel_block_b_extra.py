from global_config import *
from micro_kernel_common import get_vector_B_idx
from micro_kernel_common import get_x_B_idx
from micro_kernel_common import load_B_data_and_offset

def micro_kernel_block_b_extra(vector_id_array_B, VEC_REG_B_LEN,
                               vector_scroll_B,
                               register_scroll_B,
                               B_odd_flag,
                               LINES, COLS):
    code_str = ""
    vector_scroll_B = [i for i in range(VEC_REG_B_LEN)]
    ptr_B_POS = 0
    for j in range(VEC_REG_B_LEN):
        vector_B_idx = get_vector_B_idx(j, vector_id_array_B, vector_scroll_B)
        x_B_idx = get_x_B_idx(B_odd_flag, register_scroll_B)
        code_str_b, ptr_B_POS, B_odd_flag = load_B_data_and_offset(vector_B_idx, x_B_idx, ptr_B_POS, B_odd_flag, COLS)
        code_str += code_str_b
    return code_str, ptr_B_POS, B_odd_flag