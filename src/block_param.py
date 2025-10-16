
from global_config import *

def NRSA(N, NR_MAIN):
    CEIL_NC = (N + SIMD_LANE - 1) // SIMD_LANE # 按照SIMD宽度分块
    logger.debug(f"CEIL_NC = (N + SIMD_LANE - 1) // SIMD_LANE = ({N} + {SIMD_LANE} - 1) // {SIMD_LANE} = {CEIL_NC} (N方向需要多少个SIMD寄存器)")
    NR_REMAIN = CEIL_NC % NR_MAIN # 剩余的块数
    logger.debug(f"NR_REMAIN = CEIL_NC % NR_MAIN = {CEIL_NC} % {NR_MAIN} = {NR_REMAIN} (N方向的剩余块数)")
    NR_MAIN_LOOPS = CEIL_NC // NR_MAIN # 主循环的次数
    logger.debug(f"NR_MAIN_LOOPS = CEIL_NC // NR_MAIN = {CEIL_NC} // {NR_MAIN} = {NR_MAIN_LOOPS} (N方向的主循环的次数)")
    NR_REMAIN_LOOPS = 1 if NR_REMAIN else 0 # 剩余的块有没有循环
    logger.debug(f"NR_REMAIN_LOOPS = 1 if NR_REMAIN else 0 = 1")
    logger.debug(f"NR_REMAIN_LOOPS = 1 if {NR_REMAIN} else 0")
    logger.debug(f"NR_REMAIN_LOOPS = {NR_REMAIN_LOOPS} (N方向的剩余循环)")
    # if NR_MAIN == 3:
    #   if NR_REMAIN == 1 and NR_MAIN_LOOPS >= 1 :
    #     NR_MAIN_LOOPS -= 1
    #     NR_REMAIN = 4
    #   elif NR_REMAIN == 2 and NR_MAIN_LOOPS >= 1 :
    #     NR_MAIN_LOOPS -= 1
    #     NR_REMAIN = 5
    # elif NR_MAIN == 4 :
    #   if NR_REMAIN == 1 and NR_MAIN_LOOPS >= 1 :
    #     NR_MAIN_LOOPS -= 1
    #     NR_REMAIN = 5
    #   elif NR_REMAIN == 2 :
    #     if NR_MAIN_LOOPS >= 2 :
    #       NR_MAIN_LOOPS -= 2
    #       NR_REMAIN = 5
    #       NR_REMAIN_LOOPS = 2
    #     elif NR_MAIN_LOOPS >= 1 :
    #       NR_MAIN_LOOPS -= 1
    #       NR_REMAIN = 6
    #   elif NR_REMAIN == 3 and NR_MAIN_LOOPS >= 3 :
    #     NR_MAIN_LOOPS -= 3
    #     NR_REMAIN = 5
    #     NR_REMAIN_LOOPS = 3
    # elif NR_MAIN == 5 :
    #   if NR_REMAIN == 1 :
    #     if NR_MAIN_LOOPS >= 3 :
    #       NR_MAIN_LOOPS -= 3
    #       NR_REMAIN = 4
    #       NR_REMAIN_LOOPS = 4
    #     elif NR_MAIN_LOOPS >= 1 :
    #       NR_MAIN_LOOPS -= 1
    #       NR_REMAIN = 6
    #   elif NR_REMAIN == 2 and NR_MAIN_LOOPS >= 2 :
    #     NR_MAIN_LOOPS -= 2
    #     NR_REMAIN = 4
    #     NR_REMAIN_LOOPS = 3
    #   elif NR_REMAIN == 3 and NR_MAIN_LOOPS >= 1 :
    #     NR_MAIN_LOOPS -= 1
    #     NR_REMAIN = 4
    #     NR_REMAIN_LOOPS = 2

    logger.debug(f"Adjusted NR_MAIN_LOOPS = {NR_MAIN_LOOPS}")
    logger.debug(f"Adjusted NR_REMAIN = {NR_REMAIN}")
    logger.debug(f"Adjusted NR_REMAIN_LOOPS = {NR_REMAIN_LOOPS}")

    return NR_MAIN_LOOPS, NR_REMAIN, NR_REMAIN_LOOPS

def MRSA(M, NR):
    MR_MAIN = min(6, (SIMD_REG_NUM - max(4, NR)) // (NR + 1))
    logger.debug(f"MR_MAIN = min(6, (32 - max(4, NR)) // (NR + 1)) = min(6, (32 - max(4, {NR})) // ({NR} + 1)) = {MR_MAIN}")
    logger.debug(f"MR_MAIN = {MR_MAIN} (最小值是6（为什么？）， 32是SIMD寄存器总量，NR不足SIMD_LANE的，按照SIMD_LANE计算)")
    MR_REMAIN = M % MR_MAIN
    logger.debug(f"MR_REMAIN = M % MR_MAIN = {M} % {MR_MAIN} = {MR_REMAIN} (M方向的剩余块数)")
    MR_MAIN_LOOPS = M // MR_MAIN
    logger.debug(f"MR_MAIN_LOOPS = M // MR_MAIN = {M} // {MR_MAIN} = {MR_MAIN_LOOPS} (M方向的主循环的次数)")
    MR_REMAIN_LOOPS = 1 if MR_REMAIN else 0
    logger.debug(f"MR_REMAIN_LOOPS = 1 if MR_REMAIN else 0")
    logger.debug(f"MR_REMAIN_LOOPS = 1 if {MR_REMAIN} else 0")
    logger.debug(f"MR_REMAIN_LOOPS = {MR_REMAIN_LOOPS} (M方向的剩余循环)")
    # if MR_MAIN == 5 :
    #   if MR_REMAIN == 1 :
    #     if MR_MAIN_LOOPS >= 3 :
    #       MR_MAIN_LOOPS -= 3
    #       MR_REMAIN = 4
    #       MR_REMAIN_LOOPS = 4
    #     elif MR_MAIN_LOOPS >= 1 :
    #       MR_MAIN_LOOPS -= 1
    #       MR_REMAIN = 3
    #       MR_REMAIN_LOOPS = 2
    #   elif MR_REMAIN == 2 and MR_MAIN_LOOPS >= 2 :
    #     MR_MAIN_LOOPS -= 2
    #     MR_REMAIN = 4
    #     MR_REMAIN_LOOPS = 3
    #   elif MR_REMAIN == 3 and MR_MAIN_LOOPS >= 1 :
    #     MR_MAIN_LOOPS -= 1
    #     MR_REMAIN = 4
    #     MR_REMAIN_LOOPS = 2
    # elif MR_MAIN == 4 :
    #   if MR_REMAIN == 1 and MR_MAIN_LOOPS >= 2 :
    #     MR_MAIN_LOOPS -= 2
    #     MR_REMAIN = 3
    #     MR_REMAIN_LOOPS = 3
    #   elif MR_REMAIN == 2 and MR_MAIN_LOOPS >= 1 :
    #       MR_MAIN_LOOPS -= 1
    #       MR_REMAIN = 3
    #       MR_REMAIN_LOOPS = 2
    # elif MR_MAIN == 3 and MR_REMAIN == 1 and MR_MAIN_LOOPS >= 1 :
    #   MR_MAIN_LOOPS -= 1
    #   MR_REMAIN = 2
    #   MR_REMAIN_LOOPS = 2
    
    logger.debug(f"Adjusted MR_MAIN_LOOPS = {MR_MAIN_LOOPS}")
    logger.debug(f"Adjusted MR_REMAIN = {MR_REMAIN}")
    logger.debug(f"Adjusted MR_REMAIN_LOOPS = {MR_REMAIN_LOOPS}")
    return MR_MAIN, MR_MAIN_LOOPS, MR_REMAIN, MR_REMAIN_LOOPS

def RBSA(M, N, NR_MAIN):
    logger.debug(f"开始计算N方向的主要参数...")
    NR_MAIN_LOOPS, NR_REMAIN, NR_REMAIN_LOOPS = NRSA(N, NR_MAIN)
    logger.debug(f"开始计算N的主循环中M方向的主要参数...")
    NR_MAIN_MR_MAIN, NR_MAIN_MR_MAIN_LOOPS, NR_MAIN_MR_REMAIN, NR_MAIN_MR_REMAIN_LOOPS = MRSA(M, NR_MAIN) if NR_MAIN_LOOPS else (0,0,0,0)
    logger.debug(f"开始计算N的剩余循环中M方向的主要参数...")
    NR_REMAIN_MR_MAIN, NR_REMAIN_MR_MAIN_LOOPS, NR_REMAIN_MR_REMAIN, NR_REMAIN_MR_REMAIN_LOOPS = MRSA(M, NR_REMAIN) if NR_REMAIN_LOOPS else (0,0,0,0)

    return NR_MAIN_LOOPS, NR_REMAIN, NR_REMAIN_LOOPS, NR_MAIN_MR_MAIN, NR_MAIN_MR_MAIN_LOOPS, NR_MAIN_MR_REMAIN, NR_MAIN_MR_REMAIN_LOOPS, NR_REMAIN_MR_MAIN, NR_REMAIN_MR_MAIN_LOOPS, NR_REMAIN_MR_REMAIN, NR_REMAIN_MR_REMAIN_LOOPS