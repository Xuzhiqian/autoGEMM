from global_config import *

def get_vector_B_idx(col,
                     UNROLL_NR,
                     j,
                     vector_id_array_B,
                     vector_scroll_B):
    return vector_id_array_B[vector_scroll_B[col * UNROLL_NR + j]]

def get_vector_C_idx(line, col,
                     UNROLL_NR,
                     j,
                     COLS):
    return line * COLS + col * UNROLL_NR + j

def get_last_simd_col(col,
                      UNROLL_NR,
                      j):
    return SIMD_LANE * (col * UNROLL_NR + j)