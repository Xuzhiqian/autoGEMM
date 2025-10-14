from global_config import *

def micro_kernel_unroll_nr_init(is_last_k, COLS, WITH_BIAS_FLAG):
    UNROLL_NR = 2 # 默认N方向要展开为2次
    if COLS % 2 != 0: # 如果分块为奇数的话，则不展开
        UNROLL_NR = 1
    if is_last_k and WITH_BIAS_FLAG:
        # 如果是最后一个K且需要加偏置，则完全展开
        # 但是这个时候貌似也不是给store_c用的
        UNROLL_NR = COLS
    logger.debug(f"UNROLL_NR: {UNROLL_NR} (展开次数)")
    return UNROLL_NR