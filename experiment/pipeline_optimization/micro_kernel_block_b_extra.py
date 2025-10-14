
def micro_kernel_block_b_extra(is_last_k,
                               vector_id_array_B, VEC_REG_B_LEN,
                               vector_scroll_B,
                               register_scroll_B,
                               B_odd_flag,
                               LINES, COLS,
                               REG_BLOCK_TRANS_FLAG):
    code_str = ""
    # Extra operations ensure that Load next block B works correctly
    if REG_BLOCK_TRANS_FLAG and is_last_k and (not COLS == VEC_REG_B_LEN):
        vector_scroll_B = [i for i in range(VEC_REG_B_LEN)]
        ptr_B_POS = 0
        for j in range(VEC_REG_B_LEN):
            code_str += f"    \"ldr     q{vector_id_array_B[vector_scroll_B[j]]}, [x{register_scroll_B[B_odd_flag]}, #{(ptr_B_POS)*16}]             \\n\"\n"
            if ptr_B_POS == COLS - 1:
                ptr_B_POS = 0
                code_str += f"    \"add     x{register_scroll_B[B_odd_flag]}, x{register_scroll_B[B_odd_flag]}, x8              \\n\"\n"
                B_odd_flag ^= 1
            else:
                ptr_B_POS += 1 
    return code_str