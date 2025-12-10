from global_config import SIMD, UNROLL_LANE
import re
import tvm
from tvm import te
from tvm import autotvm
from tvm.autotvm.task import ConfigEntity
from template.tvm_extern_asm_micro_kernel import intrin_gemm_MxNxK, gemm_MxNxK_impl

from tvm.contrib import tedd
from IPython.display import display_svg

# @autotvm.template("packB")
def packB(M, N, K, bn, kn, bn_ceil, parallel):
    # cfg = autotvm.get_config()

    B = te.placeholder((K, N), name="B")
    PackedB = te.compute(
        (K // kn, N // bn, kn, bn_ceil), 
        lambda i, x, y, z: te.if_then_else(
            z < bn, B[i * kn + y, x * bn + z], 0
        ), name="PackedB"
    )
    
    packB_schedule = te.create_schedule(PackedB.op)
    bigK, bigN, littleK, littleN = packB_schedule[PackedB].op.axis
    packB_schedule[PackedB].vectorize(littleN)
    # packB_schedule[PackedB].unroll(littleK)
    # packB_schedule[PackedB].unroll(littleN)
    # packB_schedule[PackedB].parallel(bigK)
    if parallel:
        parallel_axis = packB_schedule[PackedB].fuse(bigK, bigN)
        packB_schedule[PackedB].parallel(parallel_axis)

    return packB_schedule, [B, PackedB]