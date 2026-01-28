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
from utils.tune import tune

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to run autotvm.")
    parser.add_argument("-m", type=int, required=True, help="M")
    parser.add_argument("-n", type=int, required=True, help="N")
    parser.add_argument("-k", type=int, required=True, help="K")
    parser.add_argument("-a", "--arch", default="a64fx", choices=["linux", "a64fx"], help='select architecture linux or a64fx')
    parser.add_argument("--parallel", action="store_true", help='whether parallel execute')
    parser.add_argument(
        "-s",
        "--step",
        type=int,
        required=False,
        default=2000,
        help="Step of autotvm search.",
    )
    parser.add_argument(
        "-r",
        "--record_file",
        default="matmul.log",
        type=str,
        required=False,
        help="Specify name of the file to record autotvm tuning result",
    )
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

    record_file = args.record_file
    best_record_file = args.best_record_file
    step = args.step
    parallel = args.parallel

    target = BUILD_TARGET

    logger.info(f"Start tune for M={M}, K={K}, N={N}, record_file={record_file}, best_record_file={best_record_file}, n_trial={step}, target={target}")
    tune(M, N, K, record_file, best_record_file, parallel, n_trial=step, target=target)

    autotvm.record.pick_best(record_file, best_record_file)
