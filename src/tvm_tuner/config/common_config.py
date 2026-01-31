import os
from tvm import autotvm
from utils.ncopy_runner import NCOPYLocalRunner

cc_compiler = os.environ["TVM_CC"]

measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(
            n_parallel=None,
            timeout=100
        ),
        # runner=autotvm.LocalRunner(
        #     number=100,
        #     repeat=1,
        #     timeout=300,
        #     min_repeat_ms=100,
        #     parallel_num = 4,
        # ),
        runner=NCOPYLocalRunner(
            number=1000,
            repeat=1,
            timeout=300,
            min_repeat_ms=100,
        )
    )