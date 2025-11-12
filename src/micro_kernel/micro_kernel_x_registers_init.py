from global_config import *

def micro_kernel_x_registers_init(REG_BLOCK_TRANS_FLAG, real_cols):
    code_str = ""
    logger.debug("进入了x寄存器初始化阶段...")
    code_str += "\"\\n\" // 进入了x寄存器初始化阶段...\n"
    if REG_BLOCK_TRANS_FLAG == 2:
        code_str += f"    \"mov     {A_Head}, %[A]                 \\n\" // {A_Head}存储A头指针\n"
        code_str += f"    \"add     %[B], %[B], #{real_cols * FLOAT_BYTES}                 \\n\"\n"
        code_str += f"    \"add     %[C], %[C], #{real_cols * FLOAT_BYTES}                 \\n\"\n"
        code_str += f"    \"mov     {C_Head}, %[C]                 \\n\"\n"
    code_str += f"    \"mov     {B_Head}, %[B]                   \\n\" // {B_Head}存储B头指针\n"
    if SIMD == "NEON":
        code_str += f"    \"add     {B_Head2}, %[B], %[ldb], lsl #{LEFT_OFFSET}               \\n\" // {B_Head2}存储B + FLOAT_BYTES * ldb\n"
    if SIMD == "SVE":
        code_str += f"    \"add     {B_Head2}, %[B], %[ldb], lsr #1               \\n\" // {B_Head2}存储B + FLOAT_BYTES * ldb\n" # ???
    code_str += f"    \"prfm    PLDL1KEEP, [{B_Head}, #64]              \\n\" // B矩阵预取\n"
    code_str += f"    \"prfm    PLDL1KEEP, [{B_Head2}, #64]              \\n\" // B矩阵预取\n"
    logger.debug("进入了x寄存器初始化阶段...完成")
    code_str += "\"\\n\" // 进入了x寄存器初始化阶段...完成\n"
    return code_str