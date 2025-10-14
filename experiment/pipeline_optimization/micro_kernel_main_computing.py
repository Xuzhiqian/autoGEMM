from global_config import *
from micro_kernel_common import get_vector_C_idx
from micro_kernel_common import get_last_simd_col

def micro_kernel_main_computing(line, col,
                                UNROLL_NR,
                                vector_id_array_A, VEC_REG_A_LEN,
                                vector_id_array_B,
                                vector_scroll_A,
                                vector_scroll_B,
                                A_odd_flag,
                                mod_simd_lane_loop_id,
                                is_last_k,
                                LOOP_ID,
                                LINES, COLS,
                                LOOP_K_BEGIN_FLAG, LOOP_K_END_FLAG,
                                real_lines, real_cols,
                                WITH_BIAS_FLAG):
    is_A_odd = (A_odd_flag == 1)
    code_str = ""
    if (
        LOOP_ID == 0 and
        LOOP_K_BEGIN_FLAG and
        (not WITH_BIAS_FLAG)
    ): # fmla改成fmul计算优化
        for j in range(UNROLL_NR):
            last_simd_col = get_last_simd_col(col, UNROLL_NR, j)
            if line < real_lines and last_simd_col < real_cols:
                vector_C_idx = get_vector_C_idx(line, col, UNROLL_NR, j, COLS)
                vector_B_idx = vector_id_array_B[vector_scroll_B[col * UNROLL_NR + j]]
                vector_A_idx = vector_scroll_A[A_odd_flag][line]
                code_str += f"    \"fmul    v{vector_C_idx}.4s, v{vector_B_idx}.4s, v{vector_A_idx}.s[{mod_simd_lane_loop_id}]             \\n\"\n"
        return code_str

    for j in range(UNROLL_NR):
        last_simd_col = get_last_simd_col(col, UNROLL_NR, j)
        if line < real_lines and last_simd_col < real_cols:
            actual_line = line
            if ( # 这里看起来似乎是一种line的调整
                is_A_odd and
                (
                    (is_last_k and not WITH_BIAS_FLAG) or
                    (not is_last_k and mod_simd_lane_loop_id == 3)
                )
            ):
                actual_line = (line + VEC_REG_A_LEN % real_lines) % real_lines
                # 若VEC_REG_A_LEN = real_lines，则这里等价于
                # line = line % real_lines = line
                # 若VEC_REG_A_LEN < real_lines，则这里等价于
                # line = (line + VEC_REG_A_LEN) % real_lines
            vector_C_idx = get_vector_C_idx(actual_line, col, UNROLL_NR, j, COLS)
            vector_B_idx = vector_id_array_B[vector_scroll_B[col * UNROLL_NR + j]]
            vector_A_idx = vector_scroll_A[A_odd_flag][actual_line]
            code_str += f"    \"fmla    v{vector_C_idx}.4s, v{vector_B_idx}.4s, v{vector_A_idx}.s[{mod_simd_lane_loop_id}]             \\n\"\n"
    return code_str