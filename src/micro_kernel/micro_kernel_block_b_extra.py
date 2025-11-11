from global_config import *

def micro_kernel_block_b_extra(vector_id_array_B, VEC_REG_B_LEN,
                               vector_scroll_B,
                               register_scroll_B,
                               B_odd_flag,
                               LINES, COLS):
    code_str = ""
    vector_scroll_B = [i for i in range(VEC_REG_B_LEN)]
    ptr_B_POS = 0
    for j in range(VEC_REG_B_LEN):
        vector_B_idx = vector_id_array_B[vector_scroll_B[j]]
        x_B_idx = register_scroll_B[B_odd_flag]
        if SIMD == "NEON":
            code_str += f"    \"ldr     q{vector_B_idx}, [x{x_B_idx}, #{(ptr_B_POS) * FLOAT_BYTES}]             \\n\"\n"
        if SIMD == "SVE":
            code_str += f"    \"{LD1}     z{vector_B_idx}.{VEC_SIGN}, p0/z, [x{x_B_idx}, #{ptr_B_POS}, mul vl]             \\n\"\n"
        if ptr_B_POS == COLS - 1:
            ptr_B_POS = 0
            if SIMD == "NEON":
                code_str += f"    \"add     x{x_B_idx}, x{x_B_idx}, {LDB}              \\n\"\n"
            if SIMD == "SVE":
                code_str += f"    \"add     x{x_B_idx}, x{x_B_idx}, %[ldb]              \\n\"\n"
            B_odd_flag ^= 1
        else:
            ptr_B_POS += 1 
    return code_str, ptr_B_POS, B_odd_flag