from global_config import *
from micro_kernel_common import get_vector_C_idx
from micro_kernel_common import get_last_simd_col

def micro_kernel_store_c(line, col,
                         UNROLL_NR,
                         vector_id_array_A, VEC_REG_A_LEN,
                         is_A_odd,
                         LINES, COLS,
                         real_lines, real_cols,
                         WITH_BIAS_FLAG):
    code_str = ""
    for j in range(UNROLL_NR): # 控制在列方向（N方向）的展开次数
        if line < real_lines:
            actual_line = line
            if is_A_odd and (not WITH_BIAS_FLAG):
                actual_line = (line + VEC_REG_A_LEN % real_lines) % real_lines
            vector_C_idx = get_vector_C_idx(actual_line, col, UNROLL_NR, j, COLS)
            last_simd_col = get_last_simd_col(col, UNROLL_NR, j)
            x_C_idx = RESERVED_REG_NUM + actual_line
            if last_simd_col + SIMD_LANE <= real_cols:
                # 当前位置仍然在可以直接一整个SIMD寄存器存储的位置
                code_str += f"    \"str     q{vector_C_idx}, [x{x_C_idx}], #{SIMD_BYTES}           \\n\"\n"
            else:
                # 当前位置已经需要一个个FLOAT进行存储
                for k in range(last_simd_col, real_cols):
                    code_str += f"    \"st1     {{v{vector_C_idx}.s}}[{k % SIMD_LANE}], [x{x_C_idx}], #{FLOAT_BYTES}           \\n\"\n"
    return code_str