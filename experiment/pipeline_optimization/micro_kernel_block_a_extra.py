from global_config import *

def micro_kernel_block_a_extra(A_odd_flag,
                               is_last_k,
                               VEC_REG_A_LEN,
                               vector_scroll_A,
                               LINES, COLS,
                               real_lines, real_cols,
                               next_lines, next_cols,
                               REG_BLOCK_TRANS_FLAG,
                               WITH_BIAS_FLAG):
    code_str = ""
    # Extra operations ensure that load next block A works correctly
    if REG_BLOCK_TRANS_FLAG and is_last_k and WITH_BIAS_FLAG:
        for line in range(next_lines):
            if not (A_odd_flag == 0 or line >= real_lines - VEC_REG_A_LEN % real_lines):
                code_str += f"    \"ldr     q{vector_scroll_A[0][line]}, [x{RESERVED_REG_NUM+LINES+line}], #16    \\n\"\n"
    return code_str