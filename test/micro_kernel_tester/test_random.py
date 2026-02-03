import os
import sys
import subprocess as sp
import random
import csv

# create test_path
current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_path, "..", "..")
src_path = os.path.join(project_root, "src")

# export python environment
sys.path.insert(0, src_path)

# import python code
from global_config import UNROLL_LANE

times = 100

res = []

for i in range(times):
    M = random.randint(1, 1024)
    N = random.randint(1, 1024)
    K = random.randint(1, 1024)
    lda = random.randint(K, K + 1024)
    ldb = random.randint(N, N + 1024)
    ldc = random.randint(N, N + 1024)
    UNROLL_K = random.randint(1, 4) * (2 * UNROLL_LANE)
    NR_MAIN = random.randint(3, 5)
    PIPELINE_STRATEGY_LEVEL = random.randint(0, 1)
    MRSA_FLAG = random.randint(0, 1)
    
    args = f"python {current_path}/test.py --M {M} --N {N} --K {K} --lda {lda} --ldb {ldb} --ldc {ldc} --UNROLL_K {UNROLL_K} --NR_MAIN {NR_MAIN} --PIPELINE_STRATEGY_LEVEL {PIPELINE_STRATEGY_LEVEL} --MRSA_FLAG {MRSA_FLAG}"

    p = sp.Popen(args, stdout=sp.PIPE, text=True, shell=True)
    p.wait()
    print(f"{i}: {args}")
    s = p.stdout.read()
    print(s)
    print(p.returncode)
    res.append({
        "M":M,
        "N":N,
        "K":K,
        "dtype":"fp32",
        "perf":float(s.split("GFLOPS:")[-1])
    })
    if p.returncode: # normal value is 0
        print("returncode error!")
    
with open('random_res.csv', 'w', newline='', encoding='utf-8') as f:
    dict_writer = csv.DictWriter(f, fieldnames=res[0].keys())
    
    dict_writer.writeheader()  # 写入表头
    dict_writer.writerows(res) # 一次性写入所有行