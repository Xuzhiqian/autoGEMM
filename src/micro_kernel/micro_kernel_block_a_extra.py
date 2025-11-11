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
            vector_A_idx = vector_scroll_A[0][line]
            x_A_idx = RESERVED_REG_NUM + LINES + line
            if SIMD == "NEON":
                code_str += f"    \"ldr     q{vector_A_idx}, [x{x_A_idx}], #{SIMD_BYTES}    \\n\"\n"
            if SIMD == "SVE":
                code_str += f"    \"{LD1R}     z{vector_A_idx}.{VEC_SIGN}, p0/z, [x{x_A_idx}]    \\n\"\n"
                code_str += f"    \"add     x{x_A_idx}, x{x_A_idx}, #{UNROLL_LANE * FLOAT_BYTES}    \\n\"// 使x{x_A_idx}偏移SIMD的长度\n"
    return code_str