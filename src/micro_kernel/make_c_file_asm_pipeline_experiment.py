# Generate any shape of asm micro-kernel and ensure correct
# Compile by clang++ -O3
# older version for autoGEMM small gemm
# used for pipeline experiment

# PIPELINE_STRATEGY_LEVEL
# 0 - corresponding to figure b) , base implement of ARM ASM code, corresponding listing 2
# 1 - corresponding to figure c) , use extra register pre load A , corresponding listing 3
# 2 - optimization not shown, fuse last K with store
# 3 - corresponding to figure d) , add micro-kernel level pipeline, here all autoGEMM optimize method used

import random
import string
import sys
from global_config import *
from gemm_MxKxN_impl import gemm_MxKxN_impl
from makefile_generator import makefile_generator

M = int(sys.argv[1]) # M，N，K来自命令行参数
N = int(sys.argv[2])
K = int(sys.argv[3])
UNROLL_K = int(sys.argv[4]) # K方向的循环展开次数
NR_MAIN = int(sys.argv[5]) # N方向的主循环大小
REPEAT = int(sys.argv[6]) # 测试中循环的次数
PIPELINE_STRATEGY_LEVEL = int(sys.argv[7]) # 流水优化策略等级
MRSA_FLAG = int(sys.argv[8]) # 额外的分块值微调

UNIQ_ID_LEN = 8
uniq_id = "".join(random.choices(string.ascii_uppercase, k=UNIQ_ID_LEN))
logger.debug(f"unique_id: {uniq_id} (用于作为接口名唯一性标识)")
f = open('c_file_asm.cpp','w')
logger.debug(f"调用gemm_MxKxN_impl传入M={M}, N={N}, K={K}，A，B，C都是行主序")
f.write(gemm_MxKxN_impl(M, N, K, K, N, N, uniq_id, repeat=REPEAT, pipeline_strategy_level=PIPELINE_STRATEGY_LEVEL, UNROLL_K=UNROLL_K, NR_MAIN=NR_MAIN, MRSA_FLAG = MRSA_FLAG))
f.close()

f = open('Makefile', 'w')
f.write(makefile_generator())
f.close()
