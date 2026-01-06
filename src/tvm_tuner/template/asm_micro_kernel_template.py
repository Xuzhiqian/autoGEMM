from global_config import SIMD, UNROLL_LANE, SIMD_LANE, logger
import re
import tvm
from tvm import te
from tvm import tir
from tvm import autotvm
from tvm.autotvm.task import ConfigEntity
from template.tvm_extern_asm_micro_kernel import intrin_gemm_MxNxK, gemm_MxNxK_impl
from tvm.script import tir as T

from tvm.contrib import tedd
from IPython.display import display_svg

@autotvm.template("matmul")
def matmul(M, N, K, parallel):
    print(f"matmul({M}, {N}, {K}, {parallel})")
    cfg = autotvm.get_config()

    # Tiling structure: split M/N/K into 3 axes each.
    cfg.define_split("tile_x", M, num_outputs=3)
    cfg.define_split("tile_y", N, num_outputs=3)
    cfg.define_split("tile_k", K, num_outputs=3)

    # Micro-kernel parameters used in tensorization.
    cfg.define_knob("nr_main_knob", [3, 4, 5])
    cfg.define_knob("MRSA_FLAG", [0, 1])
    cfg.define_knob("unroll_k_knob", [UNROLL_LANE * 2, UNROLL_LANE * 4, UNROLL_LANE * 8])
    cfg.define_knob("pipeline_strategy_level_knob", [0, 1, 2, 3])
    if SIMD == "NEON" :
        cfg.define_knob("padding_size", [1, 4])
    elif SIMD == "SVE" :
        cfg.define_knob("padding_size", [1, 4, 8, 16])

    cfg.define_knob("packAB", [0, 1, 2, 3]) # 0: NPA & NPB 1: PA & NPB, 2: NPA & PB, 3: PA & PB
    # cfg.define_knob("packAB", [0, 2])
    # cfg.define_knob("packAB", [0])
    # cfg.define_knob("packAB", [1])
    # cfg.define_knob("packAB", [2])
    # cfg.define_knob("packAB", [3])

    packAB = cfg["packAB"].val
    padding_size = cfg["padding_size"].val

    # Matrix "A" has a shape of (M, K).
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    k = te.reduce_axis((0, K), "k")

    if packAB == 0:
        # C = A x B
        C = te.compute(
            (M, N),
            lambda x, y: te.sum(A[x, k] * B[k, y], axis=k),
            name='C',
        )
    elif packAB == 1:
        bm = cfg["tile_x"].size[-1]
        bk = cfg["tile_k"].size[-1]
        bk_ceil = ((bk - 1) // padding_size + 1) * padding_size
        # print(f"bm = {bm}, bk = {bk}, bk_ceil = {bk_ceil}")
        # print(f"M // bm = {M // bm}, K // bk = {K // bk}")

        PackedA = te.compute(
            (M // bm, K // bk, bm, bk_ceil),
            lambda i, x, y, z: te.if_then_else(
                z < bk,
                A[i * bm + y, x * bk + z],
                0
            )
        )

        # C = PackedA x B
        C = te.compute(
            (M, N),
            lambda x, y: te.sum(PackedA[x // bm, k // bk, x % bm, k % bk] * B[k, y], axis=k),
            name='C',
        )
    elif packAB == 2:
        # Matrix "PackedB" has been pre-packed into shape of (K // kn, N // bn, K, bn_ceil), for "bn" is the innermost axis of the splited "N" dim.
        # "bn_ceil" is a padding size to store "bn" elements.
        # Note the pre-pack format is only available for inference mode, where weight matrix "B" is fixed.
        bn = cfg["tile_y"].size[-1]
        bk = cfg['tile_k'].size[-1]
        bn_ceil = ((bn - 1) // padding_size + 1) * padding_size # bn_ceil will always be the multiplier of padding_size
        # print(f"bn = {bn}, kn = {kn}, bn_ceil = {bn_ceil}")
        # print(f"K // kn = {K // kn}, N // bn = {N // bn}")

        PackedB = te.compute(
            (K // bk, N // bn, bk, bn_ceil),
            lambda i, x, y, z: te.if_then_else(
                z < bn,
                B[i * bk + y, x * bn + z],
                0
            ),
            name='PackedB'
        )

        # C = A x PackedB
        C = te.compute(
            (M, N),
            lambda x, y: te.sum(A[x, k] * PackedB[k // bk, y // bn, k % bk, y % bn], axis=k),
            name="C",
        )
    elif packAB == 3:
        bm = cfg["tile_x"].size[-1]
        bk = cfg["tile_k"].size[-1]
        bk_ceil = ((bk - 1) // padding_size + 1) * padding_size
        bn = cfg["tile_y"].size[-1]
        bn_ceil = ((bn - 1) // padding_size + 1) * padding_size
        PackedA = te.compute(
            (M // bm, K // bk, bm, bk_ceil),
            lambda i, x, y, z: te.if_then_else(
                z < bk,
                A[i * bm + y, x * bk + z],
                0
            )
        )
        PackedB = te.compute(
            (K // bk, N // bn, bk, bn_ceil),
            lambda i, x, y, z: te.if_then_else(
                z < bn,
                B[i * bk + y, x * bn + z],
                0
            ),
            name='PackedB'
        )
        # C = PackedA x PackedB
        C = te.compute(
            (M, N),
            lambda x, y: te.sum(PackedA[x // bm, k // bk, x % bm, k % bk] * PackedB[k // bk, y // bn, k % bk, y % bn], axis=k),
            name="C",
        )
    # Schedule:
    s = te.create_schedule(C.op)
    x, y = s[C].op.axis
    (k,) = s[C].op.reduce_axis

    xt, xo, xi = cfg["tile_x"].apply(s, C, x)
    yt, yo, yi = cfg["tile_y"].apply(s, C, y)
    kt, ko, ki = cfg["tile_k"].apply(s, C, k)
    # print(f"xt, xo, xi = {xt, xo, xi}")
    # print(f"yt, yo, yi = {yt, yo, yi}")
    # print(f"kt, ko, ki = {kt, ko, ki}")
    
    # Make (yi, xi, ki) the inner most axes, to be tensorized later.
    s[C].reorder(yt, kt, xt, yo, ko, xo, yi, xi, ki)

    # Let autotvm to find the best order of the 6 axes:
    cfg.define_reorder("reorder_outer", [yt, kt, xt, yo, ko, xo], "all")
    new_order = cfg["reorder_outer"].apply(s, C, [yt, kt, xt, yo, ko, xo])

    if parallel :
        # Fuse the outmost non-reducution axes.
        sibling_axes = []
        for axis in new_order:
            if axis not in [kt, ko]:
                sibling_axes.append(axis)
            else:
                break

        parallel_axis = s[C].fuse(*sibling_axes)

        assert parallel_axis is not None
        s[C].parallel(parallel_axis)

    pragma_axis = parallel_axis if parallel else xo

    # if packAB == 2:
    #     bigK, bigN, littleK, littleN = s[PackedB].op.axis
    #     s[PackedB].vectorize(littleN)

    if packAB == 1:
        s[PackedA].compute_at(s[C], xo) # packedA at xo
    elif packAB == 2:
        s[PackedB].compute_at(s[C], xo) # packedB at xo
    elif packAB == 3:
        s[PackedA].compute_at(s[C], xo) # packedA at xo
        s[PackedB].compute_at(s[C], xo) # packedB at xo

    lda, ldb, ldc = 0, 0, 0
    if packAB == 0:
        lda = K
        ldb = N
        ldc = N
    elif packAB == 1:
        lda = bk_ceil
        ldb = N
        ldc = N
    elif packAB == 2:
        lda = K
        ldb = bn_ceil
        ldc = N
    elif packAB == 3:
        lda = bk_ceil
        ldb = bn_ceil
        ldc = N

    # Inner kernel implementation for the tensorization.
    micro_kernel, uniq_id = intrin_gemm_MxNxK(
        cfg["tile_x"].size[-1],
        cfg["tile_y"].size[-1],
        cfg["tile_k"].size[-1],
        lda,
        ldb,
        ldc,
    )
    s[C].tensorize(yi, micro_kernel)
    # graph = tedd.viz_dataflow_graph(s, show_svg=True, dot_file_path=f"/home/linzuxuan/autoGEMM/autoGEMM/data/figure/{M}_{N}_{K}.dot")
    s[C].pragma(pragma_axis, "import_llvm", gemm_MxNxK_impl(
        cfg["tile_x"].size[-1],
        cfg["tile_y"].size[-1],
        cfg["tile_k"].size[-1],
        lda,
        ldb,
        ldc,
        cfg["pipeline_strategy_level_knob"].val,
        cfg["unroll_k_knob"].val,
        cfg["nr_main_knob"].val,
        cfg["MRSA_FLAG"].val,
        uniq_id
    ))

    cfg.add_flop(2 * M * N * K) 

    return s, [A, B, C]
