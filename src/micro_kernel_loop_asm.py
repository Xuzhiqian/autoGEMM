from global_config import *
from micro_kernel_x_registers_init import micro_kernel_x_registers_init
from micro_kernel_unroll_nr_init import micro_kernel_unroll_nr_init
from micro_kernel_main_computing import micro_kernel_main_computing
from micro_kernel_store_c import micro_kernel_store_c
from micro_kernel_next_block_c import micro_kernel_next_block_c
from micro_kernel_next_block_a import micro_kernel_next_block_a
from micro_kernel_next_block_b import micro_kernel_next_block_b
from micro_kernel_block_a_extra import micro_kernel_block_a_extra
from micro_kernel_block_b_extra import micro_kernel_block_b_extra

def micro_kernel_loop_asm(
    LOOP_ID, LAST_K_ID,
    LINES, COLS,
    real_lines, real_cols,
    next_lines, next_cols,
    vector_id_array_A, VEC_REG_A_LEN,
    vector_id_array_B, VEC_REG_B_LEN,
    register_scroll_B,
    LOOP_K_BEGIN_FLAG, LOOP_K_END_FLAG,
    REG_BLOCK_TRANS_FLAG,
    FMA_CALCULATE_FLAG,
    STORE_C_FLAG,
    WITH_BIAS_FLAG
):
    logger.debug(f"LOOP_ID: {LOOP_ID} (K方向的循环ID)")
    logger.debug(f"LAST_K_ID: {LAST_K_ID} (最后的K的ID)")
    logger.debug(f"LINES: {LINES} (行数)")
    logger.debug(f"COLS: {COLS} (列数)")
    logger.debug(f"real_lines: {real_lines} (实际行数)")
    logger.debug(f"real_cols: {real_cols} (实际列数)")
    logger.debug(f"next_lines: {next_lines} ()")
    logger.debug(f"next_cols: {next_cols} ()")
    logger.debug(f"vector_id_array_A: {vector_id_array_A} (A向量ID数组)")
    logger.debug(f"VEC_REG_A_LEN: {VEC_REG_A_LEN} (A向量寄存器长度)")
    logger.debug(f"vector_id_array_B: {vector_id_array_B} (B向量ID数组)")
    logger.debug(f"VEC_REG_B_LEN: {VEC_REG_B_LEN} (B向量寄存器长度)")
    logger.debug(f"register_scroll_B: {register_scroll_B} ()")
    logger.debug(f"LOOP_K_BEGIN_FLAG: {LOOP_K_BEGIN_FLAG} (K循环开始标志, 0:K方向循环起始，在beta为0时，fmla可优化为fmul, 1:非K方向循环起始)")
    logger.debug(f"LOOP_K_END_FLAG: {LOOP_K_END_FLAG} (K循环结束标志, 0:K方向循环未结束，执行x寄存器自增, 1:K方向循环结束，即执行最后一次计算后存入C后结束)")
    logger.debug(f"REG_BLOCK_TRANS_FLAG: {REG_BLOCK_TRANS_FLAG} (寄存器块转置标志，0:, 1:, 2:)")
    logger.debug(f"FMA_CALCULATE_FLAG: {FMA_CALCULATE_FLAG} (FMA计算标志, 0:不执行计算, 1:执行计算)")
    logger.debug(f"STORE_C_FLAG: {STORE_C_FLAG} (存储C标志, 0:不存储C矩阵, 1:存储C矩阵)")
    logger.debug(f"WITH_BIAS_FLAG: {WITH_BIAS_FLAG} (带有偏置标志，0:beta为0, 1:beta不为0)")

    code_str = ""
    is_last_k = (LOOP_ID == LAST_K_ID)
    # 存在几种情况
    # 1. compile_time_for_init_func_asm: LOOP_ID = LAST_K_ID = -1
    # 
    UNROLL_NR = micro_kernel_unroll_nr_init(is_last_k, COLS, WITH_BIAS_FLAG)

    A_k_loop_id_simd = LOOP_ID // UNROLL_LANE
    A_odd_flag = A_k_loop_id_simd % 2
    is_A_odd = (A_odd_flag == 1) # 在A矩阵K方向上是奇数，注意如果LOOP_ID = -1时该值也为True
    is_A_even = (A_odd_flag == 0)
    B_odd_flag = ((LOOP_ID * COLS + VEC_REG_B_LEN) // COLS) % 2
    ptr_B_POS = (LOOP_ID * COLS + VEC_REG_B_LEN) % COLS 
    mod_simd_lane_loop_id = LOOP_ID % UNROLL_LANE

    vector_scroll_A = [[], []]
    vector_scroll_A[0] = [vector_id_array_A[i] for i in range(LINES)] # 按照顺序取A的SIMD寄存器
    vector_scroll_A[1] = [vector_id_array_A[(i + real_lines) % VEC_REG_A_LEN] for i in range(real_lines)] # 在vector_id_array_A当中取real_lines个寄存器
    # VEC_REG_A_LEN = 6
    # if real_lines = 6, then vector_scroll_A = vector_id_array_A([0, 1, 2, 3, 4, 5])
    # if real_lines = 5, then vector_scroll_A = vector_id_array_A([5, 0, 1, 2, 3])
    # if real_lines = 4, then vector_scroll_A = vector_id_array_A([4, 5, 0, 1])
    # if real_lines = 3, then vector_scroll_A = vector_id_array_A([3, 4, 5])
    # if real_lines = 2, then vector_scroll_A = vector_id_array_A([2, 3])
    # if real_lines = 1, then vector_scroll_A = vector_id_array_A([1])
    # 暂时不知道这种permutation有什么作用
    logger.debug(f"vector_scroll_A[0] = {vector_scroll_A[0]}")
    logger.debug(f"vector_scroll_A[1] = {vector_scroll_A[1]}")
    vector_scroll_B = [(i + LOOP_ID * COLS) % VEC_REG_B_LEN for i in range(COLS)]
    logger.debug(f"vector_scroll_B = {vector_scroll_B}")

    # Initializes the ABC Block pointer
    if REG_BLOCK_TRANS_FLAG and is_last_k:
        code_str += micro_kernel_x_registers_init(REG_BLOCK_TRANS_FLAG, real_cols)
        B_odd_flag = 0 # ???

    for i in range(LINES * (COLS // UNROLL_NR)):
        line = i % LINES # 当前行
        col = i // LINES # 当前列
        is_last_line = (line == LINES - 1)

        if FMA_CALCULATE_FLAG:
            code_str += micro_kernel_main_computing(line, col,
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
                                                    WITH_BIAS_FLAG)

        if STORE_C_FLAG and is_last_k:
            code_str += micro_kernel_store_c(line, col,
                                             UNROLL_NR,
                                             vector_id_array_A, VEC_REG_A_LEN,
                                             VEC_REG_B_LEN,
                                             is_A_odd,
                                             LINES, COLS,
                                             real_lines, real_cols,
                                             WITH_BIAS_FLAG)

        if LOOP_K_END_FLAG and is_last_k: # K方向终止条件
            continue

        if REG_BLOCK_TRANS_FLAG and is_last_k: # 只有启用REG_BLOCK_TRANS_FLAG及最后一个K时才需要进行C矩阵的x寄存器的偏移
            code_str += micro_kernel_next_block_c(line, col,
                                                  UNROLL_NR,
                                                  LINES, COLS,
                                                  VEC_REG_A_LEN,
                                                  VEC_REG_B_LEN,
                                                  next_lines, next_cols,
                                                  WITH_BIAS_FLAG)

        code_str += micro_kernel_next_block_a(line, col,
                                              UNROLL_NR,
                                              VEC_REG_A_LEN,
                                              vector_scroll_A,
                                              A_odd_flag,
                                              mod_simd_lane_loop_id,
                                              is_last_k,
                                              LOOP_ID, LAST_K_ID,
                                              LINES, COLS,
                                              real_lines, real_cols,
                                              next_lines, next_cols,
                                              REG_BLOCK_TRANS_FLAG,
                                              WITH_BIAS_FLAG)

        if is_last_line:
            code_str_b, ptr_B_POS, B_odd_flag = micro_kernel_next_block_b(line, col,
                                                                          UNROLL_NR,
                                                                          vector_id_array_B, VEC_REG_B_LEN,
                                                                          vector_scroll_B,
                                                                          register_scroll_B,
                                                                          B_odd_flag,
                                                                          ptr_B_POS,
                                                                          is_last_k,
                                                                          LOOP_ID, LAST_K_ID,
                                                                          LINES, COLS,
                                                                          real_lines, real_cols)
            code_str += code_str_b

    if REG_BLOCK_TRANS_FLAG and is_last_k and WITH_BIAS_FLAG:
        code_str += micro_kernel_block_a_extra(is_A_odd,
                                               VEC_REG_A_LEN,
                                               vector_scroll_A,
                                               LINES, COLS,
                                               real_lines, real_cols,
                                               next_lines, next_cols)

    if REG_BLOCK_TRANS_FLAG and is_last_k and (not COLS == VEC_REG_B_LEN): # unknown constraints
        code_str += micro_kernel_block_b_extra(vector_id_array_B, VEC_REG_B_LEN,
                                               vector_scroll_B,
                                               register_scroll_B,
                                               B_odd_flag,
                                               LINES, COLS)

    return code_str