from loguru import logger
import math

# SIMD = "NEON"
SIMD = "SVE"

# PRECISION = "DOUBLE"
PRECISION = "FLOAT"
# PRECISION = "FP16"

SIMD_BITS = 128 # NEON in bits
if SIMD == "SVE":
    SIMD_BITS = 256 # SVE in bits

FLOAT_BITS = 64 # double in bits
VEC_SIGN = "d"
DATA_TYPE = "double"
LD1 = "ld1d"
LD1R = "ld1rd"
ST1 = "st1d"
TOL = "1e-5"
if PRECISION == "FLOAT":
    FLOAT_BITS = 32 # float in bits
    VEC_SIGN = "s"
    DATA_TYPE = "float"
    LD1 = "ld1w"
    LD1R = "ld1rw"
    ST1 = "st1w"
    TOL = "1e-4"
if PRECISION == "FP16":
    FLOAT_BITS = 16 # fp16 in bits
    VEC_SIGN = "h"
    DATA_TYPE = "__fp16"
    LD1 = "ld1h"
    LD1R = "ld1rh"
    ST1 = "st1h"
    TOL = "1e0"

SIMD_BYTES = SIMD_BITS // 8
FLOAT_BYTES = FLOAT_BITS // 8
SIMD_LANE = SIMD_BITS // FLOAT_BITS

RESERVED_REG_NUM = 16
if SIMD == "SVE":
    RESERVED_REG_NUM = 6
PASSING_REG_NUM = 6 # A, B, C, lda, ldb, ldc
SIMD_REG_NUM = 32

LEFT_OFFSET = int(math.log2(FLOAT_BYTES))

UNROLL_LANE = SIMD_LANE # NEON
if SIMD == "SVE":
    UNROLL_LANE = 1

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
if SIMD == "SVE":
    A_Head = "x21"  
    B_Head = "x22"
    B_Head_idx = 22
    B_Head2 = "x23"
    B_Head2_idx = 23
    C_Head = "x24"
    NR_LOOPS_REG = "x25"
    MR_MAIN_LOOPS_REG = "x26"
    Main_K_loop_times_REG = "x27"

import time
logger.remove()
logger.add(f'../data/log/{time.strftime("%Y%m%d%H", time.localtime())}.log')
