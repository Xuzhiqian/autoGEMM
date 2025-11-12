from global_config import *
from micro_kernel_common import get_vector_C_idx
from micro_kernel_common import get_last_simd_col

def micro_kernel_main_computing(line, col,
                                UNROLL_NR,
                                vector_id_array_A, VEC_REG_A_LEN,
                                vector_id_array_B, VEC_REG_B_LEN,
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
                if SIMD == "NEON":
                    code_str += f"    \"fmul    v{vector_C_idx}.{SIMD_LANE}{VEC_SIGN}, v{vector_B_idx}.{SIMD_LANE}{VEC_SIGN}, v{vector_A_idx}.{VEC_SIGN}[{mod_simd_lane_loop_id}]             \\n\"\n"
                if SIMD == "SVE":
                    vector_C_idx = VEC_REG_A_LEN + VEC_REG_B_LEN + get_vector_C_idx(line, col, UNROLL_NR, j, COLS)
                    code_str += f"    \"fmul    z{vector_C_idx}.{VEC_SIGN}, z{vector_B_idx}.{VEC_SIGN}, z{vector_A_idx}.{VEC_SIGN}             \\n\"\n"
        return code_str

    logger.debug(f"UNROLL_NR = {UNROLL_NR}")
    code_str += f"\"\\n\" // UNROLL_NR = {UNROLL_NR}\n"
    logger.debug(f"real_lines, real_cols = {real_lines, real_cols}")
    code_str += f"\"\\n\" // real_lines, real_cols = {real_lines, real_cols}\n"
    for j in range(UNROLL_NR):
        last_simd_col = get_last_simd_col(col, UNROLL_NR, j)
        logger.debug(f"last_simd_col = {last_simd_col}")
        code_str += f"\"\\n\" // last_simd_col = {last_simd_col}\n"
        if line < real_lines and last_simd_col < real_cols:
            actual_line = line
            if ( # 这里看起来似乎是一种line的调整
                is_A_odd and
                (
                    (is_last_k and not WITH_BIAS_FLAG) or
                    (not is_last_k and mod_simd_lane_loop_id == UNROLL_LANE - 1)
                )
            ):
                actual_line = (line + VEC_REG_A_LEN % real_lines) % real_lines
                # 若VEC_REG_A_LEN = real_lines，则这里等价于
                # line = line % real_lines = line
                # 若VEC_REG_A_LEN < real_lines，则这里等价于
                # line = (line + VEC_REG_A_LEN) % real_lines
            logger.debug(f"actual_line = {actual_line}")
            code_str += f"\"\\n\" // actual_line = {actual_line}\n"
            logger.debug(f"vector_scroll_A = {vector_scroll_A}")
            code_str += f"\"\\n\" // vector_scroll_A = {vector_scroll_A}\n"
            vector_C_idx = get_vector_C_idx(actual_line, col, UNROLL_NR, j, COLS)
            vector_B_idx = vector_id_array_B[vector_scroll_B[col * UNROLL_NR + j]]
            vector_A_idx = vector_scroll_A[A_odd_flag][actual_line]
            if SIMD == "NEON":
                code_str += f"    \"fmla    v{vector_C_idx}.{SIMD_LANE}{VEC_SIGN}, v{vector_B_idx}.{SIMD_LANE}{VEC_SIGN}, v{vector_A_idx}.{VEC_SIGN}[{mod_simd_lane_loop_id}]             \\n\"\n"
            if SIMD == "SVE":
                vector_C_idx = VEC_REG_A_LEN + VEC_REG_B_LEN + get_vector_C_idx(actual_line, col, UNROLL_NR, j, COLS)
                if last_simd_col + SIMD_LANE <= real_cols:
                    code_str += f"    \"fmla    z{vector_C_idx}.{VEC_SIGN}, p0/m, z{vector_B_idx}.{VEC_SIGN}, z{vector_A_idx}.{VEC_SIGN}             \\n\"\n"
                else:
                    code_str += f"    \"fmla    z{vector_C_idx}.{VEC_SIGN}, p1/m, z{vector_B_idx}.{VEC_SIGN}, z{vector_A_idx}.{VEC_SIGN}             \\n\"\n"
    return code_str