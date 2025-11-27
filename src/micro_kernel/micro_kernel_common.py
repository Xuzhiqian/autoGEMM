from global_config import *

def get_permuted_lines(real_lines,
                       VEC_REG_A_LEN):
    permuted_lines = real_lines - VEC_REG_A_LEN % real_lines
    return permuted_lines

def get_permuted_line(line,
                      real_lines,
                      VEC_REG_A_LEN):
    permuted_line = (line + VEC_REG_A_LEN % real_lines) % real_lines
    return permuted_line

def get_vector_A_idx(line,
                     A_odd_flag,
                     vector_scroll_A):
    return vector_scroll_A[A_odd_flag][line]

def get_x_A_idx(line,
                LINES):
    return RESERVED_REG_NUM + LINES + line

def get_vector_B_idx(j,
                     vector_id_array_B,
                     vector_scroll_B):
    return vector_id_array_B[vector_scroll_B[j]]

def get_x_B_idx(B_odd_flag,
                register_scroll_B):
    return register_scroll_B[B_odd_flag]

def get_vector_C_idx(line, col,
                     UNROLL_NR,
                     j,
                     COLS,
                     VEC_REG_A_LEN,
                     VEC_REG_B_LEN):
    vector_C_idx = line * COLS + col * UNROLL_NR + j
    if SIMD == "SVE":
        vector_C_idx += VEC_REG_A_LEN + VEC_REG_B_LEN
    return vector_C_idx

def get_x_C_idx(line):
    return RESERVED_REG_NUM + line

def get_simd_col(col,
                 UNROLL_NR,
                 j):
    return col * UNROLL_NR + j

def get_last_simd_col(col,
                      UNROLL_NR,
                      j):
    last_simd_col = SIMD_LANE * get_simd_col(col, UNROLL_NR, j)
    return last_simd_col

def prefetch_C_data(real_lines):
    code_str = ""
    for line in range(real_lines):
        x_C_idx = get_x_C_idx(line)
        code_str += f"    \"prfm    PSTL1KEEP, [x{x_C_idx}, #64]              \\n\" // 从x{x_C_idx}预取C矩阵数据\n"
    return code_str

def load_A_data_and_offset(vector_A_idx, x_A_idx):
    code_str = ""
    if SIMD == "NEON":
        code_str += f"    \"ldr     q{vector_A_idx}, [x{x_A_idx}], #{SIMD_BYTES}    \\n\"// 加载x{x_A_idx}处的数据到q{vector_A_idx}，偏移{SIMD_BYTES} Bytes\n"
    if SIMD == "SVE":
        code_str += f"    \"{LD1R}     z{vector_A_idx}.{VEC_SIGN}, p0/z, [x{x_A_idx}]    \\n\"// 广播x{x_A_idx}处的数据到z{vector_A_idx}\n"
        code_str += f"    \"add     x{x_A_idx}, x{x_A_idx}, #{UNROLL_LANE * FLOAT_BYTES}    \\n\"// 使x{x_A_idx}偏移{UNROLL_LANE * FLOAT_BYTES} Bytes\n"
    return code_str

def load_B_data_and_offset(vector_B_idx, x_B_idx, ptr_B_POS, B_odd_flag, COLS):
    code_str = ""
    if SIMD == "NEON":
        code_str += f"    \"ldr     q{vector_B_idx}, [x{x_B_idx}, #{(ptr_B_POS) * SIMD_BYTES}]             \\n\" // 加载x{x_B_idx} + {ptr_B_POS * SIMD_BYTES} Bytes处的数据到q{vector_B_idx}\n"
    if SIMD == "SVE":
        code_str += f"    \"{LD1}     z{vector_B_idx}.{VEC_SIGN}, p0/z, [x{x_B_idx}, #{ptr_B_POS}, mul vl]             \\n\" // 加载x{x_B_idx} + {ptr_B_POS * SIMD_BYTES} Bytes处的数据到z{vector_B_idx}\n"
    if ptr_B_POS == COLS - 1: # last col
        ptr_B_POS = 0
        if SIMD == "NEON":
            code_str += f"    \"add     x{x_B_idx}, x{x_B_idx}, {LDB}              \\n\" // 将x{x_B_idx}加上{LDB}后存入x{x_B_idx}\n"
        if SIMD == "SVE":
            code_str += f"    \"add     x{x_B_idx}, x{x_B_idx}, %[ldb]              \\n\" // 将x{x_B_idx}加上%[ldb]后存入x{x_B_idx}\n"
        B_odd_flag ^= 1
    else:
        ptr_B_POS += 1
    return code_str, ptr_B_POS, B_odd_flag

def load_C_data(vector_C_idx, x_C_idx, simd_col):
    code_str = ""
    if SIMD == "NEON":
        code_str += f"    \"ldr     q{vector_C_idx}, [x{x_C_idx}, #{simd_col * SIMD_BYTES}]           \\n\" // 加载x{x_C_idx} + {simd_col * SIMD_BYTES} Bytes处的数据到q{vector_C_idx}\n"
    if SIMD == "SVE":
        code_str += f"    \"{LD1}     z{vector_C_idx}.{VEC_SIGN}, p0/z, [x{x_C_idx}, #{simd_col}, mul vl]           \\n\" // 加载x{x_C_idx} + {simd_col * SIMD_BYTES} Bytes处的数据到z{vector_C_idx}\n"
    return code_str

def store_C_data(vector_C_idx, x_C_idx, simd_col, last_simd_col, real_cols):
    code_str = ""
    if last_simd_col + SIMD_LANE <= real_cols: # fully storing
        if SIMD == "NEON":
            code_str += f"    \"str     q{vector_C_idx}, [x{x_C_idx}], #{SIMD_BYTES}           \\n\" // 将q{vector_C_idx}的数据存储到x{x_C_idx} + \n"
        if SIMD == "SVE":
            code_str += f"    \"{ST1}     z{vector_C_idx}.{VEC_SIGN}, p0, [x{x_C_idx}, #{simd_col}, mul vl]           \\n\"\n"
    else: # partial storing
        if SIMD == "NEON":
            for k in range(last_simd_col, real_cols):
                code_str += f"    \"st1     {{v{vector_C_idx}.{VEC_SIGN}}}[{k % SIMD_LANE}], [x{x_C_idx}], #{FLOAT_BYTES}           \\n\"\n"
        if SIMD == "SVE":
            code_str += f"    \"{ST1}     z{vector_C_idx}.{VEC_SIGN}, p1, [x{x_C_idx}, #{simd_col}, mul vl]           \\n\"\n"
    return code_str

def compute_fmul(vector_A_idx, vector_B_idx, vector_C_idx, mod_simd_lane_loop_id):
    code_str = ""
    if SIMD == "NEON":
        code_str += f"    \"fmul    v{vector_C_idx}.{SIMD_LANE}{VEC_SIGN}, v{vector_B_idx}.{SIMD_LANE}{VEC_SIGN}, v{vector_A_idx}.{VEC_SIGN}[{mod_simd_lane_loop_id}]             \\n\"\n"
    if SIMD == "SVE":
        if UNROLL_LANE == 1:
            code_str += f"    \"fmul    z{vector_C_idx}.{VEC_SIGN}, z{vector_B_idx}.{VEC_SIGN}, z{vector_A_idx}.{VEC_SIGN}             \\n\"\n"
        else:
            code_str += f"    \"fmul    z{vector_C_idx}.{VEC_SIGN}, z{vector_B_idx}.{VEC_SIGN}, z{vector_A_idx}.{VEC_SIGN}[{mod_simd_lane_loop_id}]             \\n\"\n"
    return code_str

def compute_fmla(vector_A_idx, vector_B_idx, vector_C_idx, mod_simd_lane_loop_id, last_simd_col, real_cols):
    code_str = ""
    if SIMD == "NEON":
        code_str += f"    \"fmla    v{vector_C_idx}.{SIMD_LANE}{VEC_SIGN}, v{vector_B_idx}.{SIMD_LANE}{VEC_SIGN}, v{vector_A_idx}.{VEC_SIGN}[{mod_simd_lane_loop_id}]             \\n\"\n"
    if SIMD == "SVE":
        if UNROLL_LANE == 1:
            if last_simd_col + SIMD_LANE <= real_cols:
                code_str += f"    \"fmla    z{vector_C_idx}.{VEC_SIGN}, p0/m, z{vector_B_idx}.{VEC_SIGN}, z{vector_A_idx}.{VEC_SIGN}             \\n\"\n"
            else:
                code_str += f"    \"fmla    z{vector_C_idx}.{VEC_SIGN}, p1/m, z{vector_B_idx}.{VEC_SIGN}, z{vector_A_idx}.{VEC_SIGN}             \\n\"\n"
        else:
            code_str += f"    \"fmla    z{vector_C_idx}.{VEC_SIGN}, z{vector_B_idx}.{VEC_SIGN}, z{vector_A_idx}.{VEC_SIGN}[{mod_simd_lane_loop_id}]             \\n\"\n"
    return code_str
    