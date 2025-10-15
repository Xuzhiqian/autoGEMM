from loguru import logger

SIMD_BITS = 128 # NEON in bits
FLOAT_BITS = 32 # float in bits
SIMD_BYTES = SIMD_BITS // 8
FLOAT_BYTES = FLOAT_BITS // 8
SIMD_LANE = SIMD_BITS // FLOAT_BITS # SIMD的通道数，下面是NEON代码，位宽128bits，4通道，说明是float类型

RESERVED_REG_NUM = 16
SIMD_REG_NUM = 32

import time
logger.remove()
logger.add(f'../data/log/{time.strftime("%Y%m%d%H", time.localtime())}.log')
