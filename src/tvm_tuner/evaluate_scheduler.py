from global_config import *

import tvm
from tvm import te
from tvm import autotvm
from tvm.autotvm.task import ConfigEntity

import os
import sys
import random
import string
import argparse
import numpy as np

from template.asm_micro_kernel_template import matmul
from utils.evaluate import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to run evaluation of record.")
    parser.add_argument("-m", type=int, required=True, help="M")
    parser.add_argument("-n", type=int, required=True, help="N")
    parser.add_argument("-k", type=int, required=True, help="K")
    parser.add_argument("-a", "--arch", default="a64fx", choices=["linux", "a64fx"], help='select architecture linux or a64fx')
    parser.add_argument("--parallel", action="store_true", help='whether parallel execute')
    parser.add_argument(
        "-b",
        "--best_record_file",
        default="scheduler_summary.log",
        type=str,
        required=False,
        help="Specify name of the file to record autotvm tuning result",
    )
    args = parser.parse_args()

    M = args.m
    K = args.k
    N = args.n
    parallel = args.parallel
    best_record_file = args.best_record_file

    target = BUILD_TARGET
    pack_dso = True

    logger.info(f"Start evaluate for M={M}, K={K}, N={N}, record_file={best_record_file}, parallel={parallel}, pack_dso={pack_dso}, target={target}")
    evaluate(M, N, K, best_record_file, parallel, pack_dso=pack_dso, target=target)
