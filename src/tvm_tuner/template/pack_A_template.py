from global_config import SIMD, UNROLL_LANE
import re
import tvm
from tvm import te
from tvm import autotvm
from tvm.autotvm.task import ConfigEntity
from template.tvm_extern_asm_micro_kernel import intrin_gemm_MxNxK, gemm_MxNxK_impl

from tvm.contrib import tedd
from IPython.display import display_svg

# @autotvm.template("packA")
def packA(M, N, K, bm, bk, bk_ceil, parallel):
    # cfg = autotvm.get_config()

    A = te.placeholder((M, K), name="A")
    PackedA = te.compute(
        (M // bm, K // bk, bm, bk_ceil),
        lambda i, x, y, z: te.if_then_else(
            z < bk, A[i * bm + y, x * bk + z], 0
        ), name="PackedA"
    )

    packA_schedule = te.create_schedule(PackedA.op)
    bigM, bigK, littleM, littleK = packA_schedule[PackedA].op.axis
    packA_schedule[PackedA].vectorize(littleK)
    if parallel:
        parallel_axis = packA_schedule[PackedA].fuse(bigM, bigK)
        packA_schedule[PackedA].parallel(parallel_axis)

    return packA_schedule, [A, PackedA]