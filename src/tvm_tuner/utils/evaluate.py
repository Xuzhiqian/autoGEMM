import os
import json
import tvm
from tvm import te
from tvm import autotvm
from tvm.autotvm.task import ConfigEntity
from tvm.runtime.profiling import profile_function

import numpy as np
from template.asm_micro_kernel_template import matmul

from global_config import logger

def evaluate(M, N, K, record_file, parallel, pack_dso, target="llvm"):
    print(f"Applying best history of MxNxK = {M}x{N}x{K} from {record_file}")
    ctx = tvm.cpu(0)
    dtype = "float32"

    with autotvm.apply_history_best(record_file):
        with tvm.target.Target(target):
            s, arg_buf = matmul(M, N, K, parallel)
            func = tvm.build(s, arg_buf, name="OP_GEMM_%dX%dX%d" % (M, N, K), target=target)
            func_c = tvm.build(s, arg_buf, name="OP_GEMM_%dX%dX%d" % (M, N, K), target="c")
            # print(tvm.lower(s, arg_buf))
            # print(func_c)
            # print(func_c.get_source())

            a = tvm.nd.array(np.random.uniform(-1, 1, size=(M, K)).astype(dtype), ctx)
            b = tvm.nd.array(np.random.rand(K, N).astype(dtype), ctx)
            c = tvm.nd.array(np.zeros((M, N), dtype=dtype), ctx)

            workload = autotvm.task.args_to_workload(
                [M, N, K, parallel], "matmul"
            )
            tgt = tvm.target.Target.current()
            cfg = autotvm.task.DispatchContext.current.query(tgt, workload)
            logger.debug(f"best_cfg = {cfg}")

            func(a, b, c)

    # Verify results
    # expected = np.dot(a.asnumpy(), b.asnumpy())
    # np.testing.assert_allclose(c.asnumpy(), expected, rtol=1e-2, atol=1e-4)

    # Performance
    # evaluator = func.time_evaluator(func.entry_name, ctx, number=1000, min_repeat_ms=5000)
    # mean_time = evaluator(a, b, c).mean
    # gflops = 2 * M * N * K * 1e-9 / mean_time
    # print("TVM offline GFLOPS: %f, avg time: %f ms" % (gflops, mean_time * 1000))

    # Profiling
    # profiler = profile_function(
    #     func,
    #     dev=tvm.cpu(0),
    #     collectors=[],
    #     warmup_iters=1,
    #     func_name=func.entry_name
    # )
    # prof_res = profiler(a, b, c)
    # print(prof_res)

    if pack_dso:
        current_directory = os.path.dirname(os.path.abspath(__file__))
        func_path = os.path.join(current_directory, f"../../../data/tune_output/build/gemm_obj/{func.name}.o")
        func_c_path = os.path.join(current_directory, f"../../../data/tune_output/build/gemm_source/{func.name}.c")
        func.save(func_path)
        # print(func.get_source())
        with open(func_c_path, 'w') as f:
            f.write(func_c.get_source())

        static_kernel_path = os.path.join(current_directory, f"../../../data/tune_output/build/library/GEMM_{M}X{N}X{K}_kernel.a")
        op_gemm_path = os.path.join(current_directory, f"../../../data/tune_output/build/gemm_obj/OP_GEMM_{M}X{N}X{K}*.o")
        os.system(f"ar rcs {static_kernel_path} {op_gemm_path}")

        shared_kernel_path = os.path.join(current_directory, f"../../../data/tune_output/build/library/GEMM_{M}X{N}X{K}_kernel.so")
        func.export_library(shared_kernel_path)
