from global_config import *

def micro_kernel_x_registers_init(REG_BLOCK_TRANS_FLAG, real_cols):
    code_str = ""
    logger.debug("进入了x寄存器初始化阶段...")
    code_str += "\"\\n\" // 进入了x寄存器初始化阶段...\n"
    if REG_BLOCK_TRANS_FLAG == 2:
        code_str += f"    \"mov     x10, %[A]                 \\n\" // x10存储A头指针\n"
        code_str += f"    \"add     %[B], %[B], #{real_cols * FLOAT_BYTES}                 \\n\"\n"
        code_str += f"    \"add     %[C], %[C], #{real_cols * FLOAT_BYTES}                 \\n\"\n"
        code_str += f"    \"mov     x13, %[C]                 \\n\"\n"
    code_str += f"    \"mov     x11, %[B]                   \\n\" // x11存储B头指针\n"
    code_str += f"    \"add     x12, %[B], %[ldb], lsl #2               \\n\" // x12存储B + 4 * ldb\n"
    code_str += f"    \"prfm    PLDL1KEEP, [x11, #64]              \\n\" // B矩阵预取\n"
    code_str += f"    \"prfm    PLDL1KEEP, [x12, #64]              \\n\" // B矩阵预取\n"
    logger.debug("进入了x寄存器初始化阶段...完成")
    code_str += "\"\\n\" // 进入了x寄存器初始化阶段...完成\n"
    return code_str