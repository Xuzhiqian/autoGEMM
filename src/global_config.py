from loguru import logger

SIMD_BITS = 128 # NEON in bits
FLOAT_BITS = 32 # float in bits
SIMD_BYTES = SIMD_BITS // 8
FLOAT_BYTES = FLOAT_BITS // 8
SIMD_LANE = SIMD_BITS // FLOAT_BITS # SIMD的通道数，下面是NEON代码，位宽128bits，4通道，说明是float类型

RESERVED_REG_NUM = 16
PASSING_REG_NUM = 6 # A, B, C, lda, ldb, ldc
SIMD_REG_NUM = 32

# x寄存器分配
LDA = "x6"
NR_LOOPS_REG = "x7"
LDB = "x8"
LDC = "x9"
A_Head = "x10"
B_Head = "x11"
B_Head_idx = 11
B_Head2 = "x12"
B_Head2_idx = 12
C_Head = "x13"
MR_MAIN_LOOPS_REG = "x14"
Main_K_loop_times_REG = "x15"

import time
logger.remove()
logger.add(f'../data/log/{time.strftime("%Y%m%d%H", time.localtime())}.log')
