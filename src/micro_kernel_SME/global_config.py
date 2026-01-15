from loguru import logger
import math

# x寄存器分配
# can't use x0-x7
origPA = "x0"
origPB = "x1"
pC = "x2"
LDA = "x3"
LDB = "x4"
LDC = "x5"
origM = "x6"
origN = "x7"
origK = "x8"
MIN_N = "x9"
MIN_M = "x10"
counterL = "x14"
counterI = "x18"
counterJ = "x16"
pA0 = "x24"
pAt = "x29"
pB0 = "x19"
pBn = "x15"
pBt = "x11"
wbk = "x23"
pC0 = "x20"
pC1 = "x21"
pC2 = "x22"
pC3 = "x23"
TMP_PTR = "x17"
TMP_PTR1 = "x21"
TMP_CNT = "x14"
TMP_CNT_SIN = "w14"
ELE_SIZE = "x28"
OFFSET_A = "x12"
OFFSET_B = "x13"
TMP_CNT_POST = "x23"
pA_OFFSET = "x25"
pAn = "x26"
pB_OFFSET = "x27"

LD1 = "ld1w"
LDNT1 = "ldnt1w"
STNT1 = "stnt1w"

PASSING_REG_NUM = 6
SIMD_REG_NUM = 32
TOL = "1e-4"

def SAVE_REGS():
    code_str = f""
    code_str += f".align 5\n"
    code_str += f"add     sp, sp, #-(11 * 16)\n"
    code_str += f"stp     d8, d9, [sp, #(0 * 16)]\n"
    code_str += f"stp     d10, d11, [sp, #(1 * 16)]\n"
    code_str += f"stp     d12, d13, [sp, #(2 * 16)]\n"
    code_str += f"stp     d14, d15, [sp, #(3 * 16)]\n"
    code_str += f"stp     d16, d17, [sp, #(4 * 16)]\n"
    code_str += f"stp     x18, x19, [sp, #(5 * 16)]\n"
    code_str += f"stp     x20, x21, [sp, #(6 * 16)]\n"
    code_str += f"stp     x22, x23, [sp, #(7 * 16)]\n"
    code_str += f"stp     x24, x25, [sp, #(8 * 16)]\n"
    code_str += f"stp     x26, x27, [sp, #(9 * 16)]\n"
    code_str += f"stp     x28, x29, [sp, #(10 * 16)]\n"

    return code_str

def PROLOGUE(REALNAME):
    code_str = f""
    code_str += f".text;\n"
    code_str += f".p2align 2;\n"
    code_str += f".global {REALNAME};\n"
    code_str += f".type {REALNAME}, % function;\n"
    code_str += f"{REALNAME}:\n"
    return code_str

def RESTORE_REGS():
    code_str = f""
    code_str += f"ldp     d8, d9, [sp, #(0 * 16)]\n"
    code_str += f"ldp     d10, d11, [sp, #(1 * 16)]\n"
    code_str += f"ldp     d12, d13, [sp, #(2 * 16)]\n"
    code_str += f"ldp     d14, d15, [sp, #(3 * 16)]\n"
    code_str += f"ldp     d16, d17, [sp, #(4 * 16)]\n"
    code_str += f"ldp     x18, x19, [sp, #(5 * 16)]\n"
    code_str += f"ldp     x20, x21, [sp, #(6 * 16)]\n"
    code_str += f"ldp     x22, x23, [sp, #(7 * 16)]\n"
    code_str += f"ldp     x24, x25, [sp, #(8 * 16)]\n"
    code_str += f"ldp     x26, x27, [sp, #(9 * 16)]\n"
    code_str += f"ldp     x28, x29, [sp, #(10 * 16)]\n"
    code_str += f"add     sp, sp, #(11*16)\n"
    return code_str

def START_SME_FEATURE():
    code_str = f""
    code_str += f"fmov    x23, d0\n"
    code_str += f"fmov    x24, d1\n"
    code_str += f"fmov    x25, d2\n"
    code_str += f"fmov    x26, d3\n"
    code_str += f"msr     SVCRSMZA, #1\n"
    code_str += f"isb\n"
    code_str += f"fmov    d0, x23\n"
    code_str += f"fmov    d1, x24\n"
    code_str += f"fmov    d2, x25\n"
    code_str += f"fmov    d3, x26\n"
    
    return code_str

def STOP_SME_FEATURE():
    code_str = f""
    code_str += f"msr     SVCRSMZA, #0\n"
    code_str += f"isb\n"
    
    return code_str
