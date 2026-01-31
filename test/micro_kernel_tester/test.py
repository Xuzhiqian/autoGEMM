import os
import sys
import string
import argparse
import random

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--M", type=int, default=32)
parser.add_argument("--N", type=int, default=32)
parser.add_argument("--K", type=int, default=32)
parser.add_argument("--lda", type=int, default=32)
parser.add_argument("--ldb", type=int, default=32)
parser.add_argument("--ldc", type=int, default=32)
parser.add_argument("--UNROLL_K", type=int, default=4)
parser.add_argument("--NR_MAIN", type=int, default=8)
parser.add_argument("--REPEAT", type=int, default=64)
parser.add_argument("--PIPELINE_STRATEGY_LEVEL", type=int, default=0)
parser.add_argument("--MRSA_FLAG", type=int, default=0)

args = parser.parse_args()

M = args.M
N = args.N
K = args.K
lda = args.lda
ldb = args.ldb
ldc = args.ldc
UNROLL_K = args.UNROLL_K
NR_MAIN = args.NR_MAIN
REPEAT = args.REPEAT
PIPELINE_STRATEGY_LEVEL = args.PIPELINE_STRATEGY_LEVEL
MRSA_FLAG = args.MRSA_FLAG

# create test_path
current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_path, "..", "..")
src_path = os.path.join(project_root, "src")
UNIQ_ID_LEN = 8
uniq_id = "".join(random.choices(string.ascii_uppercase, k=UNIQ_ID_LEN))
test_path = os.path.join(current_path, "tmp", uniq_id)

# export python environment
micro_kernel_src_path = os.path.join(src_path, "micro_kernel")
sys.path.insert(0, src_path)
sys.path.insert(0, micro_kernel_src_path)

# import python code
from global_config import *
from generate_micro_kenrel_test import generate_micro_kenrel_test
from generate_makefile import generate_makefile

# create test_path
if os.path.exists(test_path):
    os.rrmdir(test_path)
os.makedirs(test_path, exist_ok=True)

# generate micro_kernel_test and makefile
micro_kernel_test = generate_micro_kenrel_test(M, N, K, lda, ldb, ldc, uniq_id, repeat=REPEAT, pipeline_strategy_level=PIPELINE_STRATEGY_LEVEL, UNROLL_K=UNROLL_K, NR_MAIN=NR_MAIN, MRSA_FLAG=MRSA_FLAG)
makefile = generate_makefile()

if not micro_kernel_test or not makefile:
    print("generate micro_kernel_test or makefile failed")
    sys.exit(1)

c_file_asm_cpp_path = os.path.join(test_path, f"c_file_asm.cpp")
with open(c_file_asm_cpp_path, "w") as f:
    f.write(micro_kernel_test)

makefile_path = os.path.join(test_path, "Makefile")
with open(makefile_path, "w") as f:
    f.write(makefile)

# copy test.h and timer.h to test_path
test_h_path = os.path.join(current_path, "test.h")
timer_h_path = os.path.join(current_path, "timer.h")
os.system(f"cp {test_h_path} {test_path}")
os.system(f"cp {timer_h_path} {test_path}")

# compile
os.system(f"cd {test_path} && make -s")

# run
os.system(f"cd {test_path} && ./benchmark_kernel")

# clean
os.system(f"rm -rf {test_path}")