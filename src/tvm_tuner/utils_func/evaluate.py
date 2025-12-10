import os
import json
import tvm
from tvm import te
from tvm import autotvm
from tvm.autotvm.task import ConfigEntity

import numpy as np
from template.asm_micro_kernel_template import matmul
from template.pack_B_template import packB
from template.pack_A_template import packA

from global_config import logger

def evaluate(M, N, K, record_file, parallel, pack_dso, target="llvm"):
    print(f"Applying best history of MxNxK = {M}x{N}x{K} from {record_file}")
    ctx = tvm.cpu(0)
    dtype = "float32"

    with autotvm.apply_history_best(record_file):
        with tvm.target.Target(target):
            s, arg_buf = matmul(M, N, K, parallel)
            func = tvm.build(s, arg_buf, name="OP_GEMM_%dX%dX%d" % (M, N, K), target=target)
            logger.debug(tvm.lower(s, arg_buf))

            a = tvm.nd.array(np.random.uniform(-1, 1, size=(M, K)).astype(dtype), ctx)
            b = tvm.nd.array(np.random.rand(K, N).astype(dtype), ctx)
            c = tvm.nd.array(np.zeros((M, N), dtype=dtype), ctx)

            workload = autotvm.task.args_to_workload(
                [M, N, K, parallel], "matmul"
            )
            tgt = tvm.target.Target.current()
            cfg = autotvm.task.DispatchContext.current.query(tgt, workload)
            logger.debug(f"best_cfg = {cfg}")

            packAB = cfg["packAB"].val

            if packAB == 0:
                func(a, b, c)
            elif packAB == 1:
                padding_size = cfg["padding_size"].val
                bm = cfg["tile_x"].size[-1]
                bk = cfg["tile_k"].size[-1]
                bk_ceil = ((bk - 1) // padding_size + 1) * padding_size
                logger.debug(f"bm = {bm}, bk = {bk}, bk_ceil = {bk_ceil}")

                packed_A = tvm.nd.array(np.zeros((M // bm, K // bk, bm, bk_ceil), dtype=dtype), ctx)

                packA_schedule, packA_args = packA(M, N, K, bm, bk, bk_ceil, parallel)
                packA_func = tvm.build(packA_schedule, packA_args, name="OP_GEMM_%dX%dX%d_packA" % (M, N, K), target=target)
                logger.debug(tvm.lower(packA_schedule, packA_args))

                packA_func(a, packed_A)
                func(packed_A, b, c)
            elif packAB == 2:
                padding_size = cfg["padding_size"].val
                bn = cfg["tile_y"].size[-1]
                kn = cfg["tile_k"].size[-1]
                bn_ceil = ((bn - 1) // padding_size + 1) * padding_size
                logger.debug(f"bn = {bn}, kn = {kn}, bn_ceil = {bn_ceil}")

                packed_b = tvm.nd.array(np.zeros((K // kn, N // bn, kn, bn_ceil), dtype=dtype), ctx)

                packB_schedule, packB_args = packB(M, N, K, bn, kn, bn_ceil, parallel)
                packB_func = tvm.build(packB_schedule, packB_args, name="OP_GEMM_%dX%dX%d_packB" % (M, N, K), target=target)
                logger.debug(tvm.lower(packB_schedule, packB_args))

                packB_func(b, packed_b)
                func(a, packed_b, c)

    # Verify results
    expected = np.dot(a.asnumpy(), b.asnumpy())
    np.testing.assert_allclose(c.asnumpy(), expected, rtol=1e-2, atol=1e-4)

    # Performance
    # evaluator = func.time_evaluator(func.entry_name, ctx, number=1000, min_repeat_ms=5000)
    # mean_time = evaluator(a, packed_b, c).mean
    # mean_time = evaluator(a, b, c).mean
    # gflops = 2 * M * N * K * 1e-9 / mean_time
    # print("TVM offline GFLOPS: %f, avg time: %f ms" % (gflops, mean_time * 1000))

    pack_func = None
    if packAB == 1:
        pack_func = packA_func
    elif packAB == 2:
        pack_func = packB_func

    if pack_dso:
        current_directory = os.path.dirname(os.path.abspath(__file__))
        func_path = os.path.join(current_directory, f"../../../data/tune_output/build/gemm_obj/{func.name}.o")
        func.save(func_path)
        if pack_func:
            pack_func_path = os.path.join(current_directory, f"../../../data/tune_output/build/gemm_obj/{pack_func.name}.o")
            pack_func.save(pack_func_path)

        static_kernel_path = os.path.join(current_directory, f"../../../data/tune_output/build/library/GEMM_{M}X{N}X{K}_kernel.a")
        op_gemm_path = os.path.join(current_directory, f"../../../data/tune_output/build/gemm_obj/OP_GEMM_{M}X{N}X{K}*.o")
        os.system(f"ar rcs {static_kernel_path} {op_gemm_path}")

        shared_kernel_path = os.path.join(current_directory, f"../../../data/tune_output/build/library/GEMM_{M}X{N}X{K}_kernel.so")
        if pack_func:
            func.import_module(pack_func)
        func.export_library(shared_kernel_path)
