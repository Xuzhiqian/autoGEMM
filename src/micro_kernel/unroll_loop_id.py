from global_config import *

def UNROLL_LOOP_ID(K, UNROLL_K):
    BEGIN_LOOP_ID = 1
    EDGE_BEGIN_LOOP_ID = 1
    # 原始代码
    if K % UNROLL_K > UNROLL_LANE: # 如果K不是UNROLL_K的整数倍，并且剩下的K方向的数量要比SIMD_LANE大
        EDGE_BEGIN_LOOP_ID = (K % UNROLL_K) - (K % UNROLL_LANE) # 计算边界开始循环的位置
    elif K % UNROLL_K == 0 and UNROLL_K > UNROLL_LANE: # 如果K是UNROLL_K的整数倍，并且UNROLL_K比SIMD_LANE要大
        EDGE_BEGIN_LOOP_ID = UNROLL_K - UNROLL_LANE
    
    # # 新代码
    # if UNROLL_K <= SIMD_LANE:
    #     EDGE_BEGIN_LOOP_ID = 1
    #     return BEGIN_LOOP_ID, EDGE_BEGIN_LOOP_ID

    # if K % UNROLL_K >= SIMD_LANE:
    #     EDGE_BEGIN_LOOP_ID = UNROLL_K - SIMD_LANE

    return BEGIN_LOOP_ID, EDGE_BEGIN_LOOP_ID

    # 由于K方向是有循环展开的，循环展开的数量就是UNROLL_K，那么每个循环内所计算的K方向就需要UNROLL_K // SIMD_LANE个SIMD寄存器
    # K = 11, UNROLL_K = 8
    # K % UNROLL_K = 3, K % SIMD_LANE = 3
    # EDGE_BEGIN_LOOP_ID = 1
    # K = 12, UNROLL_K = 8
    # K % UNROLL_K = 4, K % SIMD_LANE = 0
    # EDGE_BEGIN_LOOP_ID = 1
    # K = 13, UNROLL_K = 8
    # K % UNROLL_K = 5, K % SIMD_LANE = 1
    # EDGE_BEGIN_LOOP_ID = 5 - 1 = 4
    # K = 14, UNROLL_K = 8
    # K % UNROLL_K = 6, K % SIMD_LANE = 2
    # EDGE_BEGIN_LOOP_ID = 6 - 2 = 4
    # K = 15, UNROLL_K = 8
    # K % UNROLL_K = 7, K % SIMD_LANE = 3
    # EDGE_BEGIN_LOOP_ID = 7 - 3 = 4
    # K = 16, UNROLL_K = 8
    # K % UNROLL_K = 0, UNROLL_K = 8 > SIMD_LANE = 4
    # EDGE_BEGIN_LOOP_ID = 8 - 4 = 4

    # if UNROLL_K == SIMD_LANE:
    #   EDGE_BEGIN_LOOP_ID = 1
    # if UNROLL_K < SIMD_LANE:
    #   EDGE_BEGIN_LOOP_ID = 1

    # 注意到BEGIN_LOOP_ID恒等于1
    # 从上面可以观察到
    # 当K可以被UNROLL_K整除时，EDGE_BEGIN_LOOP_ID就是UNROLL_K - SIMD_LANE
