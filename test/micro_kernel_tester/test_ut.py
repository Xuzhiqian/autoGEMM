import os
import sys
import subprocess as sp
import shutil

# create test_path
current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_path, "..", "..")
src_path = os.path.join(project_root, "src")

# export python environment
sys.path.insert(0, src_path)

# import python code
from global_config import UNROLL_LANE

def is_valid_combination(pairs):
    M, N, K, lda, ldb, ldc = pairs

    if lda < K or ldb < N or ldc < N:
        return False

    return True

case_count = 0
cases = []
# read from testcases_ut.csv (has header, start from second line)
# for i, pairs in enumerate(itertools.product(*parameters)): # 完全组合
for i, line in enumerate(open(os.path.join(current_path, "testcases_ut.csv"))):
    if i == 0:
        continue

    line = line.strip()
    if not line:
        continue

    pairs = list(map(int, line.split(",")))

    # check if the combination is valid
    if not is_valid_combination(pairs):
        continue

    M, N, K, lda, ldb, ldc = pairs

    for UNROLL_K in [(2 * UNROLL_LANE), (2 * UNROLL_LANE) * 2, (2 * UNROLL_LANE) * 3, (2 * UNROLL_LANE) * 4]:
        for NR_MAIN in [3, 4, 5]:
            for PIPELINE_STRATEGY_LEVEL in [0, 1, 2, 3]:
                for MRSA_FLAG in [0, 1]:
                    args = f"python {current_path}/test.py --M {M} --N {N} --K {K} --lda {lda} --ldb {ldb} --ldc {ldc} --UNROLL_K {UNROLL_K} --NR_MAIN {NR_MAIN} --PIPELINE_STRATEGY_LEVEL {PIPELINE_STRATEGY_LEVEL} --MRSA_FLAG {MRSA_FLAG}"

                    case_count += 1
                    p = sp.Popen(args, stdout=sp.PIPE, text=True, shell=True)
                    cases.append((case_count, args, p))

print(f"Total case count: {case_count}")

failed_case_count = 0
for case in cases:
    i, args, result = case
    result.wait()
    print(f"{i}: {args}")
    print(result.stdout.read())
    print(result.returncode)
    if result.returncode: # normal value is 0
        failed_case_count += 1
        print("returncode error!")
    print()

print(f"Failed case count: {failed_case_count}")
print(f"Pass rate: {(case_count - failed_case_count) / case_count * 100}%")