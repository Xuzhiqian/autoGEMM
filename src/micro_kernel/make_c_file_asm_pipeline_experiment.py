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
from gemm_MxNxK_impl import gemm_MxNxK_impl
from makefile_generator import makefile_generator

M = int(sys.argv[1]) # M，N，K来自命令行参数
N = int(sys.argv[2])
K = int(sys.argv[3])
lda = int(sys.argv[4])
ldb = int(sys.argv[5])
ldc = int(sys.argv[6])
UNROLL_K = int(sys.argv[7]) # K方向的循环展开次数
NR_MAIN = int(sys.argv[8]) # N方向的主循环大小
REPEAT = int(sys.argv[9]) # 测试中循环的次数
PIPELINE_STRATEGY_LEVEL = int(sys.argv[10]) # 流水优化策略等级
MRSA_FLAG = int(sys.argv[11]) # 额外的分块值微调
UNIQ_ID = sys.argv[12] # 接口唯一性标识
TEST_PATH = sys.argv[13] # 测试路径

current_directory = os.path.dirname(os.path.abspath(__file__))
c_file_asm_cpp_path = os.path.join(TEST_PATH, f'c_file_asm.cpp')
f = open(c_file_asm_cpp_path,'w')
logger.debug(f"Calling gemm_MxNxK_impl(M={M}, N={N}, K={K}，lda={lda}，ldb={ldb}，ldc={ldc}, uniq_id={UNIQ_ID})")
f.write(gemm_MxNxK_impl(M, N, K, lda, ldb, ldc, UNIQ_ID, repeat=REPEAT, pipeline_strategy_level=PIPELINE_STRATEGY_LEVEL, UNROLL_K=UNROLL_K, NR_MAIN=NR_MAIN, MRSA_FLAG=MRSA_FLAG))
f.close()

makefile_path = os.path.join(TEST_PATH, f'Makefile')
f = open(makefile_path, 'w')
f.write(makefile_generator())
f.close()
