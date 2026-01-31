from global_config import *
from m_dim_func_asm import m_dim_func_asm
from compile_time_for_init_func_asm import compile_time_for_init_func_asm
from compile_time_for_loop_k_end_func_asm import compile_time_for_loop_k_end_func_asm
from compile_time_for_n_dim_micro_kernel_pipeline_func_asm import compile_time_for_n_dim_micro_kernel_pipeline_func_asm

class RegisterManager:
    
    def __init__(self, K, MR_MAIN, NR, pipeline_strategy_level):
        self.K = K
        self.MR_MAIN = MR_MAIN
        self.NR = NR
        self.pipeline_strategy_level = pipeline_strategy_level
        self._MIN_N_REGISTER = 4
        self._MIN_N_UNROLL_REGISTER = 8
        self._NR_UNROLL_SIGN = 6
        # default value (ideal situation)
        self.VEC_REG_A_LEN = self.MR_MAIN
        self.VEC_REG_B_LEN = self.NR

    def get_vec_reg_B_len(self):
        # if self.K > SIMD_LANE * self._MIN_N_REGISTER:
        #     self.VEC_REG_B_LEN = max(self._MIN_N_REGISTER, self.NR)
        # if self.NR == self._NR_UNROLL_SIGN:
        #     if self.K > SIMD_LANE * self._MIN_N_UNROLL_REGISTER:
        #         self.VEC_REG_B_LEN = self._MIN_N_UNROLL_REGISTER

        logger.debug(f"VEC_REG_B_LEN: {self.VEC_REG_B_LEN}")
        return self.VEC_REG_B_LEN

    def get_vec_reg_C_len(self):
        self.VEC_REG_C_LEN = self.VEC_REG_A_LEN * self.VEC_REG_B_LEN
        logger.debug(f"VEC_REG_C_LEN: {self.VEC_REG_C_LEN}")
        return self.VEC_REG_C_LEN

    def get_vec_reg_A_len(self):
        # REMAIN_REG_FOR_A = SIMD_REG_NUM - self.VEC_REG_C_LEN - self.VEC_REG_B_LEN
        # if self.pipeline_strategy_level >= 1:
        #     if self.K > SIMD_LANE * self._MIN_N_REGISTER:
        #         self.VEC_REG_A_LEN = min(REMAIN_REG_FOR_A, 2 * self.MR_MAIN)

        logger.debug(f"VEC_REG_A_LEN: {self.VEC_REG_A_LEN}")
        return self.VEC_REG_A_LEN

def get_vector_id_array_A(VEC_REG_C_LEN, VEC_REG_B_LEN, VEC_REG_A_LEN):
    vector_id_array_A = [i for i in range(VEC_REG_C_LEN + VEC_REG_B_LEN, VEC_REG_C_LEN + VEC_REG_B_LEN + VEC_REG_A_LEN)]
    if SIMD == "SVE":
        vector_id_array_A = [i for i in range(VEC_REG_A_LEN)]
    logger.debug(f"vector_id_array_A: {vector_id_array_A} (A矩阵的{VEC_REG_A_LEN}个寄存器的编号)")
    return vector_id_array_A

def get_vector_id_array_B(VEC_REG_C_LEN, VEC_REG_B_LEN, VEC_REG_A_LEN):
    vector_id_array_B = [i for i in range(VEC_REG_C_LEN, VEC_REG_C_LEN + VEC_REG_B_LEN)]
    if SIMD == "SVE":
        vector_id_array_B = [i for i in range(VEC_REG_A_LEN, VEC_REG_A_LEN + VEC_REG_B_LEN)]
    logger.debug(f"vector_id_array_B: {vector_id_array_B} (B矩阵的{VEC_REG_B_LEN}个寄存器的编号)")
    return vector_id_array_B

def get_register_scroll_B():
    register_scroll_B = [B_Head_idx, B_Head2_idx]
    logger.debug(f"register_scroll_B: {register_scroll_B} (B矩阵的两个x寄存器，后面看到是在交叉地使用这两个寄存器进行B矩阵的数据加载)")
    return register_scroll_B

def n_dim_func_asm(
  REMAIN_N,
  K, UNROLL_K,
  NR, NR_LOOPS,
  MR_MAIN, MR_MAIN_LOOPS,
  MR_REMAIN, MR_REMAIN_LOOPS,
  with_bias,
  pipeline_strategy_level
):
    logger.debug(f"进入N方向的函数生成...")
    logger.debug(f"MR_MAIN: {MR_MAIN}")
    logger.debug(f"NR: {NR}")
    register_manager = RegisterManager(K, MR_MAIN, NR, pipeline_strategy_level)
    VEC_REG_B_LEN = register_manager.get_vec_reg_B_len()
    VEC_REG_C_LEN = register_manager.get_vec_reg_C_len()
    VEC_REG_A_LEN = register_manager.get_vec_reg_A_len()
    assert VEC_REG_A_LEN + VEC_REG_B_LEN + VEC_REG_C_LEN <= SIMD_REG_NUM
    assert VEC_REG_A_LEN <= 8 # fmla constraint

    vector_id_array_A = get_vector_id_array_A(VEC_REG_C_LEN, VEC_REG_B_LEN, VEC_REG_A_LEN)
    vector_id_array_B = get_vector_id_array_B(VEC_REG_C_LEN, VEC_REG_B_LEN, VEC_REG_A_LEN)
    register_scroll_B = get_register_scroll_B()

    Main_N_flag = (NR_LOOPS > 1) or (SIMD_LANE * NR == REMAIN_N) # NR_LOOPS > 1说明是N方向的NR_MAIN_LOOPS, NR_REMAIN_LOOPS最大就是1；NR_MAIN_LOOPS = 1时SIMD_LANE * NR_MAIN = min(N, SIMD_LANE * NR_MAIN * NR_MAIN_LOOPS); 而NR_REMAIN_LOOPS = 1时是下面的这种关系，即SIMD_LANE * NR_REMAIN > N - SIMD_LANE * NR_MAIN * NR_MAIN_LOOPS
    Edge_N_flag = SIMD_LANE * NR * NR_LOOPS > REMAIN_N
    Edge_N = REMAIN_N % (SIMD_LANE * NR) # 重新计算出N - SIMD_LANE * NR_MAIN * NR_MAIN_LOOPS
    if Edge_N_flag:
        NR_LOOPS -= 1 # 由于NR_LOOPS = NR_REMAIN_LOOPS = 1，所以这里被赋值为0，这句貌似是一句废代码，因为Edge_N_flag分支当中根本不用到NR_LOOPS
    if SIMD == "SVE":
        Main_N_flag = 0 if NR_LOOPS == 0 else 1
    # Main_N_flag和Edge_N_flag是否是互斥关系，即Edge_N_flag必为!Main_N_flag ?

    lines_branch_1 = MR_MAIN if MR_MAIN_LOOPS else MR_REMAIN
    lines_branch_2 = MR_MAIN if not MR_REMAIN_LOOPS else MR_REMAIN
    cols_branch_1 = SIMD_LANE * NR if Main_N_flag else Edge_N
    cols_branch_2 = SIMD_LANE * NR if not Edge_N_flag else Edge_N

    code_str = f""

    logger.debug(f"Main_N_flag: {Main_N_flag} (是否是N的主循环)")
    logger.debug(f"Edge_N_flag: {Edge_N_flag} (是否是N的剩余循环)")
    logger.debug(f"Edge_N: {Edge_N} (剩余循环所需处理的长度)")
    logger.debug(f"NR_LOOPS: {NR_LOOPS} ()")
    logger.debug(f"lines_branch_1: {lines_branch_1} (M方向的参数，如果是M方向主循环，则为MR_MAIN)")
    logger.debug(f"lines_branch_2: {lines_branch_2} (M方向的参数，如果不是M方向剩余循环，则为MR_MAIN)")
    code_str += f"\"\\n\" // lines_branch_1: {lines_branch_1}\n"
    code_str += f"\"\\n\" // lines_branch_2: {lines_branch_2}\n"
    logger.debug(f"cols_branch_1: {cols_branch_1} (N方向的参数， 如果是N方向主循环，则为SIMD_LANE * NR)")
    logger.debug(f"cols_branch_2: {cols_branch_2} (N方向的参数， 如果不是N方向剩余循环，则为SIMD_LANE * NR)")
    code_str += f"\"\\n\" // cols_branch_1: {cols_branch_1}\n"
    code_str += f"\"\\n\" // cols_branch_2: {cols_branch_2}\n"
    # 这几个参数的使用也不太明确，下面用到的调用有
    # compile_time_for_init_func_asm
    # compile_time_for_n_dim_micro_kernel_pipeline_func_asm
    # compile_time_for_loop_k_end_func_asm
    
    code_str += f"\"\\n\" // SVE版本对p寄存器的设置...\n"
    if SIMD == "SVE":
        code_str += f"    \"ptrue     p0.{VEC_SIGN}                  \\n\"\n"
        code_str += f"    \"mov       x28, #{SIMD_LANE if REMAIN_N % SIMD_LANE == 0 else REMAIN_N % SIMD_LANE}                  \\n\"\n"
        code_str += f"    \"whilelt   p1.{VEC_SIGN}, xzr, x28                  \\n\"\n"
    code_str += f"\"\\n\" // SVE版本对p寄存器的设置...完成\n"
    code_str += compile_time_for_init_func_asm(
        MR_MAIN, NR, 
        lines_branch_1, cols_branch_1,
        vector_id_array_A, VEC_REG_A_LEN,
        vector_id_array_B, VEC_REG_B_LEN,
        register_scroll_B,
        with_bias
    )

    if Main_N_flag : # Enter the N-dim main operation
        logger.debug(f"进入了N方向的主循环...")
        code_str += "\"\\n\" // 进入了N方向的主循环...\n"
        if NR_LOOPS > 1 : # Cyclic N-dim main operation
            logger.debug(f"进入了NR_LOOPS循环...")
            code_str += "\"\\n\" // 进入了NR_LOOPS循环...\n"
            code_str += f"    \"mov     {NR_LOOPS_REG}, #{NR_LOOPS}                   \\n\" // 进入NR_LOOPS循环\n"
            logger.debug(f"{NR_LOOPS_REG}当中存了NR_LOOPS，并且在一次循环后会减1，这个就是N方向的主循环")
            code_str += f"    \"b       6f                                 \\n\"\n"
            logger.debug(f"跳回到6处，6代表的是")
            code_str += f"  \"0:                                 \\n\"\n"
            code_str += f"    \"subs    {NR_LOOPS_REG}, {NR_LOOPS_REG}, #1                            \\n\"\n"
            code_str += f"    \"beq     7f                       \\n\"\n"
            code_str += compile_time_for_n_dim_micro_kernel_pipeline_func_asm(
                MR_MAIN, NR,
                K, UNROLL_K,
                lines_branch_2, SIMD_LANE * NR,
                lines_branch_1, SIMD_LANE * NR,
                vector_id_array_A, VEC_REG_A_LEN,
                vector_id_array_B, VEC_REG_B_LEN,
                register_scroll_B,
                with_bias,
                pipeline_strategy_level
            )
            code_str += f"  \"6:                                 \\n\"\n"
            logger.debug(f"进入了NR_LOOPS循环...完成")
            code_str += "\"\\n\" // 进入了NR_LOOPS循环...完成\n"
        logger.debug(f"进入了NR_LOOPS上的最后一次LOOP...")
        code_str += "\"\\n\" // 进入了NR_LOOPS上的最后一次LOOP...\n"
        code_str += m_dim_func_asm(
            MR_MAIN, MR_MAIN_LOOPS,
            MR_REMAIN, MR_REMAIN_LOOPS,
            NR,
            K, UNROLL_K,
            SIMD_LANE * NR,
            vector_id_array_A, VEC_REG_A_LEN,
            vector_id_array_B, VEC_REG_B_LEN,
            register_scroll_B,
            with_bias,
            pipeline_strategy_level
        )
        logger.debug(f"进入了NR_LOOPS上的最后一次LOOP...完成")
        code_str += "\"\\n\" // 进入了NR_LOOPS上的最后一次LOOP...完成\n"

        if NR_LOOPS > 1 : 
            logger.debug(f"NR_LOOPS循环结尾的位置，跳回到0处进行NR_LOOPS变量的递减以及再一次的LOOP")
            code_str += f"    \"b       0b                                 \\n\"\n"
            code_str += f"  \"7:                                 \\n\"\n"
        logger.debug(f"进入了N方向的主循环...完成")
        code_str += "\"\\n\" // 进入了N方向的主循环...完成\n"

    if Edge_N_flag : # Enter the N-dim remain operation
        logger.debug(f"进入了N方向的剩余循环...")
        code_str += "\"\\n\" // 进入了N方向的剩余循环...\n"
        if Main_N_flag : # Cyclic N-dim remain operation
            code_str += compile_time_for_n_dim_micro_kernel_pipeline_func_asm(
                MR_MAIN, NR,
                K, UNROLL_K,
                lines_branch_2, SIMD_LANE * NR,
                lines_branch_1, Edge_N,
                vector_id_array_A, VEC_REG_A_LEN,
                vector_id_array_B, VEC_REG_B_LEN,
                register_scroll_B,
                with_bias,
                pipeline_strategy_level
            )
        code_str += m_dim_func_asm(
            MR_MAIN, MR_MAIN_LOOPS,
            MR_REMAIN, MR_REMAIN_LOOPS,
            NR,
            K, UNROLL_K,
            Edge_N,
            vector_id_array_A, VEC_REG_A_LEN,
            vector_id_array_B, VEC_REG_B_LEN,
            register_scroll_B,
            with_bias,
            pipeline_strategy_level
        )
        logger.debug(f"进入了N方向的剩余循环...完成")
        code_str += "\"\\n\" // 进入了N方向的剩余循环...完成\n"

    code_str += compile_time_for_loop_k_end_func_asm(
        MR_MAIN, NR,
        K, UNROLL_K,
        lines_branch_2, cols_branch_2,
        vector_id_array_A, VEC_REG_A_LEN,
        vector_id_array_B, VEC_REG_B_LEN,
        register_scroll_B,
        pipeline_strategy_level
    )

    return code_str