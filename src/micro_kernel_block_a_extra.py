from global_config import *

def micro_kernel_block_a_extra(is_A_odd,
                               VEC_REG_A_LEN,
                               vector_scroll_A,
                               LINES, COLS,
                               real_lines, real_cols,
                               next_lines, next_cols):
    code_str = ""
    for line in range(next_lines):
        if is_A_odd and line < real_lines - VEC_REG_A_LEN % real_lines:
            x_A_idx = RESERVED_REG_NUM + LINES + line
            code_str += f"    \"ldr     q{vector_scroll_A[0][line]}, [x{x_A_idx}], #{SIMD_BYTES}    \\n\"\n"
    return code_str