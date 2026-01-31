from global_config import *
from block_param import RBSA
from n_dim_func_asm import n_dim_func_asm


def laf_asm_code(M: int, N: int, K: int, lda: int, ldb: int, ldc: int, pipeline_strategy_level: int=0, UNROLL_K: int=8, NR_MAIN: int=4, MRSA_FLAG: int=0, with_bias: int=0) -> str:
    """The core generator interface of autoGEMM

    Args:
        M (int): M
        N (int): N
        K (int): K
        lda (int): lda
        ldb (int): ldb
        ldc (int): ldc
        pipeline_strategy_level (int, optional): micro-kernel pipeline strategy optimization level
        UNROLL_K (int, optional): unrolling size on K direction. Defaults to 8.
        NR_MAIN (int, optional): register number used on N direction. Defaults to 4.
        MRSA_FLAG (int, optional): enable m direction dynamic tiling. Defaults to 0.
        with_bias (int, optional): if 0, then C = AB, else C = AB + C. Defaults to 0.

    Returns:
        code_str (str): generated micro-kernel. An empty string will be returned if any error occurs, otherwise it should be a completed micro-kernel.

    """    

    # UNROLL_K是有默认值8的，刚好就是SIMD_LANE * 2
    # NR_MAIN默认值是4，初步认为其含义是N方向的一个块的长度需要多少个SIMD寄存器，例如这里4的话，说明N方向的大小是128*4bits
    # with_bias含义比较明显，就是考虑beta不等于0
    code_str = "" # 主要的拼接逻辑，拼接出来的kernel就是解决整个大的输入的M，N，K的small_gemm接口

    logger.debug(f"UNROLL_K: {UNROLL_K}")
    logger.debug(f"UNROLL_LANE: {UNROLL_LANE}")
    if UNROLL_K % (2 * UNROLL_LANE) != 0: # UNROLL_K必须是2倍UNROLL_LANE的整数倍
        logger.error(f"UNROLL_K should be times of {2 * UNROLL_LANE}!")
        return code_str
    if UNROLL_K < 4:
        logger.error(f"UNROLL_K should not be smaller than 4!")
        return code_str
    logger.debug(f"NR_MAIN: {NR_MAIN}")
    if NR_MAIN not in {3, 4, 5}: # NR_MAIN限定是3、4、5中的值
        logger.error(f"NR_MAIN should be 3 or 4 or 5, not {NR_MAIN}!")
        return code_str

    logger.debug(f"M: {M}, N: {N}, K: {K}, lda: {lda}, ldb: {ldb}, ldc: {ldc}")
    if lda < K:
        logger.error(f"lda should not be smaller than K!")
        return code_str
    if ldb < N:
        logger.error(f"ldb should not be smaller than N!")
        return code_str
    if ldc < N:
        logger.error(f"ldc should not be smaller than N!")
        return code_str
    
    logger.debug(f"MRSA_FLAG: {MRSA_FLAG}, with_bias: {with_bias}")
    if MRSA_FLAG not in {0, 1}:
        logger.error(f"MRSA_FLAG should be 0 or 1, not {MRSA_FLAG}!")
        return code_str

    logger.debug(f"调用RBSA进行分块参数的计算...")
    NR_MAIN_LOOPS, NR_REMAIN, NR_REMAIN_LOOPS, NR_MAIN_MR_MAIN, NR_MAIN_MR_MAIN_LOOPS, NR_MAIN_MR_REMAIN, NR_MAIN_MR_REMAIN_LOOPS, NR_REMAIN_MR_MAIN, NR_REMAIN_MR_MAIN_LOOPS, NR_REMAIN_MR_REMAIN, NR_REMAIN_MR_REMAIN_LOOPS = RBSA(M, N, NR_MAIN, MRSA_FLAG) # 拆解出各个小块参数的逻辑（参见论文中的Fig5的d图）
    logger.debug(f"调用RBSA进行分块参数的计算...完毕")

    logger.debug(f"NR_MAIN_LOOPS: {NR_MAIN_LOOPS} (N方向是否要进行主循环)")
    logger.debug(f"NR_REMAIN: {NR_REMAIN} (N方向剩余的块数)")
    logger.debug(f"NR_REMAIN_LOOPS: {NR_REMAIN_LOOPS} (N方向是否要进行剩余循环)")
    logger.debug(f"NR_MAIN_MR_MAIN_LOOPS: {NR_MAIN_MR_MAIN_LOOPS} (M方向是否要进行主循环)")
    code_str += f"// NR_MAIN_MR_MAIN_LOOPS: {NR_MAIN_MR_MAIN_LOOPS}\n"
    logger.debug(f"NR_MAIN_MR_MAIN: {NR_MAIN_MR_MAIN} (M方向主循环的次数)")
    code_str += f"// NR_MAIN_MR_MAIN: {NR_MAIN_MR_MAIN}\n"
    logger.debug(f"NR_MAIN_MR_REMAIN_LOOPS: {NR_MAIN_MR_REMAIN_LOOPS} (M方向是否要进行剩余循环)")
    code_str += f"// NR_MAIN_MR_REMAIN_LOOPS: {NR_MAIN_MR_REMAIN_LOOPS}\n"
    logger.debug(f"NR_MAIN_MR_REMAIN: {NR_MAIN_MR_REMAIN} (M方向剩余的块数)")
    code_str += f"// NR_MAIN_MR_REMAIN: {NR_MAIN_MR_REMAIN}\n"
    logger.debug(f"NR_REMAIN_MR_MAIN_LOOPS: {NR_REMAIN_MR_MAIN_LOOPS} (N方向剩余循环中是否要在M方向上进行主循环)")
    logger.debug(f"NR_REMAIN_MR_MAIN: {NR_REMAIN_MR_MAIN} (N方向剩余循环中在M方向上的主循环次数)")
    logger.debug(f"NR_REMAIN_MR_REMAIN_LOOPS: {NR_REMAIN_MR_REMAIN_LOOPS} (N方向剩余循环中是否要在M方向上进行剩余循环)")
    logger.debug(f"NR_REMAIN_MR_REMAIN: {NR_REMAIN_MR_REMAIN} (N方向剩余循环中是否要在M方向上的剩余块数)")

    logger.debug(f"开始进行small_gemm (with bias={with_bias}) 的拼接...")
    code_str += f""
    code_str += f"asm volatile(\n"
    code_str += "\"\\n\" // 进入了整个small_gemm的初始化...\n"
    code_str += f"    \"prfm    PLDL1KEEP, [%[A], #64]     \\n\" // A矩阵预取\n"
    code_str += f"    \"prfm    PLDL1KEEP, [%[B], #64]     \\n\" // B矩阵预取\n"
    if SIMD == "NEON":
        code_str += f"    \"lsl     {LDA}, %[lda], #{LEFT_OFFSET}             \\n\" // {LDA}存储lda乘以FLOAT_BYTES，方便后面做偏移\n"
        code_str += f"    \"lsl     {LDB}, %[ldb], #{LEFT_OFFSET + 1}             \\n\" // {LDB}存储ldb乘以FLOAT_BYTES再乘以2（跳两行）\n"
        code_str += f"    \"lsl     {LDC}, %[ldc], #{LEFT_OFFSET}             \\n\" // {LDC}存储ldc乘以FLOAT_BYTES\n"
    if SIMD == "SVE":
        code_str += f"    \"lsl     %[lda], %[lda], #{LEFT_OFFSET}             \\n\" // {LDA}存储lda乘以FLOAT_BYTES，方便后面做偏移\n"
        code_str += f"    \"lsl     %[ldb], %[ldb], #{LEFT_OFFSET + 1}             \\n\" // {LDB}存储ldb乘以FLOAT_BYTES再乘以2（跳两行）\n"
        code_str += f"    \"lsl     %[ldc], %[ldc], #{LEFT_OFFSET}             \\n\" // {LDC}存储ldc乘以FLOAT_BYTES\n"
    code_str += f"    \"mov     {A_Head}, %[A]                  \\n\" // {A_Head}存储A头指针\n"
    code_str += f"    \"mov     {C_Head}, %[C]                  \\n\" // {C_Head}存储C头指针\n"
    code_str += "\"\\n\" // 进入了整个small_gemm的初始化...完成\n"

    if NR_MAIN_LOOPS: # 主循环，对B、C矩阵来说就是处理了SIMD_LANE * NR_MAIN * sizeof(float)的宽度
        code_str += "\"\\n\" // 进入了N方向主循环...\n"
        code_str += n_dim_func_asm(
            min(N, SIMD_LANE * NR_MAIN * NR_MAIN_LOOPS), # 在N和SIMD_LANE * NR_MAIN * NR_MAIN_LOOPS中取最小值，因为可能N大于SIMD_LANE * NR_MAIN * NR_MAIN_LOOPS
            K, UNROLL_K,
            NR_MAIN, NR_MAIN_LOOPS,
            NR_MAIN_MR_MAIN, NR_MAIN_MR_MAIN_LOOPS,
            NR_MAIN_MR_REMAIN, NR_MAIN_MR_REMAIN_LOOPS,
            with_bias,
            pipeline_strategy_level
        )
        code_str += "\"\\n\" // 进入了N方向主循环...完成\n"

    if NR_REMAIN_LOOPS:
        code_str += "\"\\n\" // 进入了N方向剩余循环...\n"
        if NR_MAIN_LOOPS:
            code_str += "\"\\n\" // 处理N方向主循环和剩余循环之间的变化...\n"
            code_str += f"    \"mov     {A_Head}, %[A]                 \\n\" // {A_Head}恢复为A头指针\n"
            code_str += f"    \"add     %[B], %[B], #{SIMD_LANE * NR_MAIN * FLOAT_BYTES}                 \\n\" // # B指针跳到SIMD_LANE * NR_MAIN * FLOAT_BYTES的位置\n" 
            code_str += f"    \"add     %[C], %[C], #{SIMD_LANE * NR_MAIN * FLOAT_BYTES}                 \\n\" // # C指针跳到SIMD_LANE * NR_MAIN * FLOAT_BYTES的位置\n"
            code_str += f"    \"mov     {C_Head}, %[C]                 \\n\" // {C_Head}恢复为C头指针\n"
            code_str += "\"\\n\" // 处理N方向主循环和剩余循环之间的变化...完成\n"
        code_str += n_dim_func_asm(
            N - SIMD_LANE * NR_MAIN * NR_MAIN_LOOPS, # 剩余循环所要处理的N
            K, UNROLL_K,
            NR_REMAIN, NR_REMAIN_LOOPS,
            NR_REMAIN_MR_MAIN, NR_REMAIN_MR_MAIN_LOOPS,
            NR_REMAIN_MR_REMAIN, NR_REMAIN_MR_REMAIN_LOOPS,
            with_bias,
            pipeline_strategy_level
        )
        code_str += "\"\\n\" // 进入了N方向剩余循环...完成\n"

    code_str += ""
    code_str += ": [A]\"=r\"(A),\n"
    code_str += "  [B]\"=r\"(B),\n"
    code_str += "  [C]\"=r\"(C),\n"
    code_str += "  [lda]\"=r\"(lda),\n"
    code_str += "  [ldb]\"=r\"(ldb),\n"
    code_str += "  [ldc]\"=r\"(ldc) \n"
    code_str += ": \"0\"(A),\n"
    code_str += "  \"1\"(B),\n"
    code_str += "  \"2\"(C),\n"
    code_str += "  \"3\"(lda),\n"
    code_str += "  \"4\"(ldb),\n"
    code_str += "  \"5\"(ldc) \n"
    # code_str += "  [lda]\"r\"(lda),\n"
    # code_str += "  [ldb]\"r\"(ldb),\n"
    # code_str += "  [ldc]\"r\"(ldc) \n"
    code_str += ": \"cc\", \"memory\"\n"
    code_str += "  "
    # for i in range(PASSING_REG_NUM, RESERVED_REG_NUM + 2 * max(NR_MAIN_MR_MAIN, NR_REMAIN_MR_MAIN)): # 使用到的x寄存器
    for i in range(PASSING_REG_NUM, 29):
        code_str += f", \"x{i}\""
    code_str += f"\n"
    for i in range(SIMD_REG_NUM):
        if SIMD == "NEON":
            code_str += f", \"v{i}\""
        if SIMD == "SVE":
            code_str += f", \"z{i}\""
    code_str +=  f"""
  );
"""
    return code_str