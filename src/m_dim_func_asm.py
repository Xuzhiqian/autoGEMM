from global_config import *
from compile_time_for_loop_k_remain_func_asm import compile_time_for_loop_k_remain_func_asm
from compile_time_for_m_dim_micro_kernel_pipeline_func_asm import compile_time_for_m_dim_micro_kernel_pipeline_func_asm
from compile_time_for_loop_k_begin_func_asm import compile_time_for_loop_k_begin_func_asm
from compile_time_for_loop_k_main_body_func_asm import compile_time_for_loop_k_main_body_func_asm

def m_dim_func_asm(
    MR_MAIN, MR_MAIN_LOOPS,
    MR_REMAIN, MR_REMAIN_LOOPS,
    NR,
    K, UNROLL_K,
    real_cols,
    vector_id_array_A, VEC_REG_A_LEN,
    vector_id_array_B, VEC_REG_B_LEN,
    register_scroll_B,
    with_bias,
    pipeline_strategy_level
):

    Main_K_loop_flag = (K > UNROLL_K) # K > UNROLL_K进行loop
    Main_K_loop_times = (K + UNROLL_K - 1) // UNROLL_K # K除以UNROLL_K向上取整

    code_str = f""

    if MR_MAIN_LOOPS: # Enter the M-dim main operation
        logger.debug("进入了M方向的主循环...")
        code_str += "\"\\n\" // 进入了M方向的主循环...\n"
        if MR_MAIN_LOOPS > 1 : # Cyclic M-dim main operation
            logger.debug("进入了M方向的主循环的主要操作...")
            code_str += f"    \"mov     {MR_MAIN_LOOPS_REG}, #{MR_MAIN_LOOPS}                   \\n\" // {MR_MAIN_LOOPS_REG}存储MR_MAIN_LOOPS的值\n"
            code_str += f"    \"b       1f                                      \\n\" // 跳到1\n"
            code_str += f"  \"2:                                 \\n\" // K方向的剩余操作\n"
            code_str += f"    \"subs    {MR_MAIN_LOOPS_REG}, {MR_MAIN_LOOPS_REG}, #1                            \\n\"\n"
            code_str += compile_time_for_loop_k_remain_func_asm(
                MR_MAIN, NR,
                K, UNROLL_K,
                MR_MAIN, real_cols,
                vector_id_array_A, VEC_REG_A_LEN,
                vector_id_array_B, VEC_REG_B_LEN,
                register_scroll_B
            )
            code_str += f"    \"beq     3f              \\n\" // 等于0的话跳到3, 下方是M方向的主要操作\n"
            code_str += compile_time_for_m_dim_micro_kernel_pipeline_func_asm(
                MR_MAIN, NR,
                K, UNROLL_K,
                MR_MAIN, real_cols,
                MR_MAIN, real_cols,
                vector_id_array_A, VEC_REG_A_LEN,
                vector_id_array_B, VEC_REG_B_LEN,
                register_scroll_B,
                with_bias,
                pipeline_strategy_level
            )
            code_str += f"  \"1:                                 \\n\" // K方向的主要操作\n"
            logger.debug("进入了M方向的主循环的主要操作...完成")

        # K-dim main operation
        if Main_K_loop_flag: 
            logger.debug("进入了K方向的主循环的主要操作...")
            code_str += f"    \"mov     {Main_K_loop_times_REG}, #{Main_K_loop_times}                   \\n\" // {Main_K_loop_times_REG}存储K方向的循环次数Main_K_loop_times\n"
            code_str += f"    \"subs    {Main_K_loop_times_REG}, {Main_K_loop_times_REG}, #1                            \\n\"\n"
            code_str += compile_time_for_loop_k_begin_func_asm(
                MR_MAIN, NR,
                K, UNROLL_K,
                MR_MAIN, real_cols,
                vector_id_array_A, VEC_REG_A_LEN,
                vector_id_array_B, VEC_REG_B_LEN,
                register_scroll_B,
                with_bias
            )
            code_str += f"    \"b       4f                                 \\n\" // 跳到4\n"
            code_str += f"  \"5:                                 \\n\"\n"
            code_str += compile_time_for_loop_k_main_body_func_asm(
                MR_MAIN, NR,
                K, UNROLL_K,
                MR_MAIN, real_cols,
                vector_id_array_A, VEC_REG_A_LEN,
                vector_id_array_B, VEC_REG_B_LEN,
                register_scroll_B
            )
            code_str += f"  \"4:                                 \\n\"\n"
            if MR_MAIN_LOOPS > 1 :
                code_str += f"    \"beq     2b                       \\n\" // 等于0的话跳到2\n"
            else:
                code_str += f"    \"beq     3f                       \\n\" // 等于0的话跳到3\n"
            code_str += f"    \"subs    {Main_K_loop_times_REG}, {Main_K_loop_times_REG}, #1                            \\n\" // Main_K_loop_times -= 1\n"
            code_str += f"    \"b       5b                                 \\n\" // 跳转到5\n"
            logger.debug("进入了K方向的主循环的主要操作...完成")
        else:
            logger.debug("进入了K方向的主循环的起始操作...")
            code_str += compile_time_for_loop_k_begin_func_asm(
                MR_MAIN, NR,
                K, UNROLL_K,
                MR_MAIN, real_cols,
                vector_id_array_A, VEC_REG_A_LEN,
                vector_id_array_B, VEC_REG_B_LEN,
                register_scroll_B,
                with_bias
            )
            if MR_MAIN_LOOPS > 1 :
                code_str += f"    \"b       2b                       \\n\"\n"
            logger.debug("进入了K方向的主循环的起始操作...完成")

        if MR_MAIN_LOOPS > 1 or Main_K_loop_flag:
            code_str += f"  \"3:                                 \\n\"\n"

        if not MR_MAIN_LOOPS > 1 :
            logger.debug("进入了K方向的主循环的剩余操作...")
            code_str += compile_time_for_loop_k_remain_func_asm(
                MR_MAIN, NR,
                K, UNROLL_K,
                MR_MAIN, real_cols,
                vector_id_array_A, VEC_REG_A_LEN,
                vector_id_array_B, VEC_REG_B_LEN,
                register_scroll_B
            )
            logger.debug("进入了K方向的主循环的剩余操作...完成")

        logger.debug("进入了M方向的主循环...完成")
        code_str += "\"\\n\" // 进入了M方向的主循环...完成\n"

    if MR_REMAIN_LOOPS: # Enter the M-dim remain operation
        logger.debug("进入了M方向的剩余循环...")
        if MR_MAIN_LOOPS: # Cyclic M-dim remain operation
            code_str += compile_time_for_m_dim_micro_kernel_pipeline_func_asm(
                MR_MAIN, NR,
                K, UNROLL_K,
                MR_MAIN, real_cols,
                MR_REMAIN, real_cols,
                vector_id_array_A, VEC_REG_A_LEN,
                vector_id_array_B, VEC_REG_B_LEN,
                register_scroll_B,
                with_bias
            )
        if MR_REMAIN_LOOPS > 1:
            code_str += f"    \"mov     {MR_MAIN_LOOPS_REG}, #{MR_REMAIN_LOOPS}                   \\n\"\n"
            code_str += f"    \"b       1f                                 \\n\"\n"
            code_str += f"  \"2:                                 \\n\"\n"
            code_str += f"    \"subs    {MR_MAIN_LOOPS_REG}, {MR_MAIN_LOOPS_REG}, #1                            \\n\"\n"
            code_str += compile_time_for_loop_k_remain_func_asm(
                MR_MAIN, NR,
                K, UNROLL_K,
                MR_REMAIN, real_cols,
                vector_id_array_A, VEC_REG_A_LEN,
                vector_id_array_B, VEC_REG_B_LEN,
                register_scroll_B
            )
            code_str += f"    \"beq     3f              \\n\"\n"
            code_str += compile_time_for_m_dim_micro_kernel_pipeline_func_asm(
                MR_MAIN, NR,
                K, UNROLL_K,
                MR_REMAIN, real_cols,
                MR_REMAIN, real_cols,
                vector_id_array_A, VEC_REG_A_LEN,
                vector_id_array_B, VEC_REG_B_LEN,
                register_scroll_B,
                with_bias,
                pipeline_strategy_level
            )
            code_str += f"  \"1:                                 \\n\"\n"

        # K-dim main operation
        if Main_K_loop_flag: 
            code_str += f"    \"mov     {Main_K_loop_times_REG}, #{Main_K_loop_times}                   \\n\"\n"
            code_str += f"    \"subs    {Main_K_loop_times_REG}, {Main_K_loop_times_REG}, #1                            \\n\"\n"
            code_str += compile_time_for_loop_k_begin_func_asm(
                MR_MAIN, NR,
                K, UNROLL_K,
                MR_REMAIN, real_cols,
                vector_id_array_A, VEC_REG_A_LEN,
                vector_id_array_B, VEC_REG_B_LEN,
                register_scroll_B,
                with_bias
            )
            code_str += f"    \"b       4f                                 \\n\"\n"
            code_str += f"  \"5:                                 \\n\"\n"
            code_str += compile_time_for_loop_k_main_body_func_asm(
                MR_MAIN, NR,
                K, UNROLL_K,
                MR_REMAIN, real_cols,
                vector_id_array_A, VEC_REG_A_LEN,
                vector_id_array_B, VEC_REG_B_LEN,
                register_scroll_B
            )
            code_str += f"  \"4:                                 \\n\"\n"
            if MR_REMAIN_LOOPS > 1 :
                code_str += f"    \"beq     2b                       \\n\"\n"
            else:
                code_str += f"    \"beq     3f                       \\n\"\n"
            code_str += f"    \"subs    {Main_K_loop_times_REG}, {Main_K_loop_times_REG}, #1                            \\n\"\n"
            code_str += f"    \"b       5b                                 \\n\"\n"
        else:
            code_str += compile_time_for_loop_k_begin_func_asm(
                MR_MAIN, NR,
                K, UNROLL_K,
                MR_REMAIN, real_cols,
                vector_id_array_A, VEC_REG_A_LEN,
                vector_id_array_B, VEC_REG_B_LEN,
                register_scroll_B,
                with_bias
            )
            if MR_REMAIN_LOOPS > 1 :
                code_str += f"    \"b       2b                       \\n\"\n"

        if MR_REMAIN_LOOPS > 1 or Main_K_loop_flag:
            code_str += f"  \"3:                                 \\n\"\n"

        if not MR_REMAIN_LOOPS > 1 :
            code_str += compile_time_for_loop_k_remain_func_asm(
                MR_MAIN, NR,
                K, UNROLL_K,
                MR_REMAIN, real_cols,
                vector_id_array_A, VEC_REG_A_LEN,
                vector_id_array_B, VEC_REG_B_LEN,
                register_scroll_B
            )

        logger.debug("进入了M方向的剩余循环...完成")

    return code_str
